"""Compressed sensing main script"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
print('Tensorflow version', tf.__version__)
from functools import partial
from skimage.io import imread
from matplotlib import pyplot as plt
from datetime import datetime


# == decoder part == #
def _pad(x, kernel_size, pad):
    to_pad = int((kernel_size - 1) / 2)
    if to_pad > 0 and pad == 'reflection':
        return tf.pad(x, ((0, 0), (to_pad, to_pad), (to_pad, to_pad), (0, 0)), mode='REFLECT')
    else:
        return x

def _upsample(x, upsample_mode):
    if upsample_mode == 'none':
        return x
    
    else:
        w = tf.shape(x)[1]
        h = tf.shape(x)[2]
        new_shape = tf.stack([w * 2, h * 2], axis=0)
        try:
            # align_corners = True necessary for bilinear interpolation
            x = tf.image.resize_images(x, new_shape, align_corners=True,
                                       method=getattr(tf.image.ResizeMethod, upsample_mode.upper()))
            return x
        except AttributeError:
            raise NotImplementedError('%s rescaling' % upsample_mode)
            
def _bn(x, bn_affine):
    return tf.layers.batch_normalization(x, trainable=bn_affine, momentum=0.9, epsilon=1e-5, training=True)      

def decodernw(inputs,
              num_output_channels=3, 
              num_channels_up=[128] * 5,
              filter_size_up=1,
              need_sigmoid=True, 
              pad='reflection',
              upsample_mode='bilinear', 
              act_fun=tf.nn.relu, # tf.nn.leaky_relu 
              bn_before_act=False,
              bn_affine=True,
              upsample_first=True
             ):
    """Deep Decoder.
       Takes as inputs a 4D Tensor (batch, width, height, channels)""" #which is the fixed noise with channels = 
    ## Configure
    num_channels_up = num_channels_up[1:] + [num_channels_up[-1], num_channels_up[-1]] #[128, 128 ,...] with shape of [1,6]
    n_scales = len(num_channels_up) #equals to 6
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up] * n_scales #[1, 1 ,...] with shape of [1,6]                                               
        
    ## Deep Decoder
    net = inputs
    for i, (num_channels, kernel_size) in enumerate(zip(num_channels_up, filter_size_up)):       
        # Upsample (first)
        if upsample_first and i != 0:
            net = _upsample(net, upsample_mode)

        # Conv        
        net = _pad(net, kernel_size, pad)
        net = tf.layers.conv2d(net, num_channels, kernel_size=kernel_size, strides=1, activation=None, padding='valid', use_bias=False)

        # Upsample (second)
        if not upsample_first and i != len(num_channels_up) - 1:
            net = _upsample(net, upsample_mode)

        # Batch Norm + activation
        if bn_before_act: 
            net = _bn(net, bn_affine)           
        net = act_fun(net)
        if not bn_before_act: 
            net = _bn(net, bn_affine) 
                
    # Final convolution
    kernel_size = 1
    net = _pad(net, kernel_size, pad)
    net = tf.layers.conv2d(net, num_output_channels, kernel_size, strides=1, 
                           activation=None, padding='valid', use_bias=False)
    if need_sigmoid:
        net = tf.nn.sigmoid(net)    
    return net #Outputs a 4D Tensor (batch, width*2^6, height*2^6, channels=3) 


# == fit part == #
def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables(scope="DeepDecoder"):
        params = 1
        for dim in variable.get_shape():
            params *= dim.value
        total_parameters += params
    return total_parameters

def fit(net,
        num_channels,
        img_shape,
        num_measurements,
        y_feed,
        A_feed,
        reg_noise_std=0,
        reg_noise_decayevery=100000,
        reg_noise_decay_rate=0.7,
        lr_decay_epoch=0,
        lr_decay_rate=0.65,
        LR=0.01,
        OPTIMIZER='adam',
        num_iter=5000,
        find_best=False,
        verbose=False,
        device='gpu'
       ):
    """Fit a model.
    
        Args: 
        net: the generative model
        num_channels: Number of upsample channels #e.g.[k, k ,...] with shape of [1,6]
        img_shape: original real image shape, a 4D tensor, e.g. [1,64,64,3] 
        num_measurements: the number of measurements 
        y_feed, A_feed: real oberservation y and measurment matrix A
        reg_noise_std, reg_noise_decayevery, reg_noise_decay_rate: parameters of the random noise to input z 
        LR, lr_decay_epoch, lr_decay_rate: parameters of learning rate 
        device: device name 
        
    """
    
    with tf.Graph().as_default():
        # Global step
        global_step = tf.train.get_or_create_global_step()
            
        with tf.device('/%s' % device):  
            # Set up palceholders
            n_input = img_shape[1]*img_shape[2]*img_shape[3]
            A = tf.placeholder(tf.float32, shape=(n_input, num_measurements), name='A') #e.g.[img_wid*img_high*3, img_wid*img_high*3]
            y = tf.placeholder(tf.float32, shape=(1, num_measurements), name='y') #e.g.[1, img_wid*img_high*3]
            
            # Define input uniform noise 
            totalupsample = 2**len(num_channels) #e.g.2^6
            width = int(img_shape[1] / totalupsample)
            height = int(img_shape[2] / totalupsample)
            z = tf.constant(np.random.uniform(size=[1, width, height, num_channels[0]]).astype(np.float32) * 1. / 10, name='z') 
            z_saved = z
            
            # Optional random noise to inputs
            if reg_noise_std > 0:
                reg_noise = tf.train.exponential_decay(reg_noise_std, global_step, reg_noise_decayevery, reg_noise_decay_rate, staircase=True)
                z += tf.random_uniform(tf.shape(z)) * reg_noise
            
            # Deep decoder prior 
            feed_forward = tf.make_template("DeepDecoder", net) #feed_forward takes a 4D Tensor (batch, width, height, channels) as input and outputs a 4D Tensor (batch, width*2^6, height*2^6, channels=3)
            x = feed_forward(z) #net_output with shape [1, img_wid, img_high, 3] 
            x_saved = feed_forward(z_saved) #original output image with shape [1, img_wid, img_high, 3]
            
            # Compressed sensing         
            y_hat = tf.matmul(x.reshape(1,-1), A, name='y_hat')
            y_hat_from_saved = tf.matmul(x_saved.reshape(1,-1), A, name='y_hat_from_saved')
        
            # Define loss  
            mse = tf.losses.mean_squared_error
            loss = mse(y, y_hat)

            # Define learning rate 
            if lr_decay_epoch > 0:
                LR = tf.train.exponential_decay(LR, global_step, lr_decay_epoch, lr_decay_rate, staircase=True)

            # Define optimizer 
            if OPTIMIZER == 'SGD':
                print("optimize with SGD", LR)
                optimizer = torch.optim.GradientDescentOptimizer(LR, 0.9)
            elif OPTIMIZER == 'adam':
                print("optimize with adam", LR)
                optimizer = tf.train.AdamOptimizer(LR)
            elif OPTIMIZER == 'LBFGS':
                raise NotImplementedError('LBFGS Optimizer')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)    

            # Additional output information
            if reg_noise_std > 0: 
                true_loss_original = mse(y, y_hat_from_saved)

        with tf.Session() as sess:
            # Init            
            mse = [0.] * num_iter
            mse_original = [0.] * num_iter
            sess.run(tf.global_variables_initializer())    
            
            if find_best:
                if not os.path.exists('log'):
                    os.makedirs('log/')
                save_log_dir = os.path.join('log', datetime.now().strftime("%m-%d_%H-%M"))
                if verbose:
                    print('Save net in', save_log_dir)
                saver = tf.train.Saver(max_to_keep=1)
                saver.save(sess, os.path.join(save_log_dir, 'net'), global_step=0)
                best_mse = 1000000.0
                best_img = sess.run(x_saved)
            
            # Feed dict
            feed_dict = {A: A_feed, y: y_feed}
            
            # Optimize
            num_params = get_num_params()
            sess.graph.finalize()
            print('\x1b[37mFinal graph size: %.2f MB\x1b[0m' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))

            for i in range(num_iter):
                if reg_noise_std <= 0:  
                    loss_, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    true_loss_original_ = loss_
                else:
                    loss_, true_loss_original_, _ = sess.run([loss, true_loss_original, train_op], feed_dict=feed_dict)
                mse[i] = loss_
                mse_original[i] = true_loss_original_
                        
                # Display
                if i > 0 and i % 10 == 0:
                    print ('\r[Iteration %05d] loss=%.5f  true loss orig=%.5f' % (i, loss_, true_loss_original_), end='')  
                
                # Best net
                if find_best and best_mse > 1.005 * loss_:
                    best_mse = loss_
                    best_img = sess.run(x_saved)
                    saver.save(sess, os.path.join(save_log_dir, 'net'), global_step=i + 1)                  
                        
            # Return final image or best found so far if `find_best`
            if find_best:
                out_img = best_img
                print()
                print('Best MSE (wrt noisy)', best_mse)
            else:
                out_img = sess.run(x_saved)
            if verbose:
                return sess.run(z_saved), out_img, num_params
            else:
                return sess.run(z_saved), out_img


# == main part == #
def load_img(path, img_name):
    img_path = os.path.join(path, img_name)
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = img[None, :, :, :]
    img_clean = img / 255.
    return img_clean
    
def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""
    if A is None:
        y_hat = x_hat
    else:
        y_hat = np.matmul(x_hat, A)
    assert y_hat.shape == y.shape
    return np.mean((y - y_hat) ** 2)
    
def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)
 
def main(hparams):

    # Get inputs
    x_real = load_img(hparams.path, hparams.img_name)
    
    # Construct measurements
    mea_shape = x_real.data.shape[1] * x_real.data.shape[2] * x_real.data.shape[3] #i.e. img_wid*img_wid*3
    if hparams.type_measurements == 'random':
        A = tf.constant(np.random.randn(mea_shape, hparams.num_measurements).astype(np.float32), name='measurement_input')
        noise_shape = [1, num_measurements]
    elif hparams.type_measurements == 'identity':
        A = tf.constant(numpy.identity(mea_shape).astype(np.float32), name='measurement_input')
        noise_shape = [1, mea_shape]
    
    # Construct oberservation
    observ_noise = noise_std * np.random.randn(size=noise_shape)
    y_real = np.matmul(x_real, A) + observ_noise #[10,100]
    
    # Define num_channles 
    num_channels = [hparams.k]*hparams.num_channel
    
    # Define decoder network 
    net_fn = partial(decodernw, 
                     num_output_channels=x_real.shape[-1], 
                     num_channels_up=num_channels, 
                     upsample_first=False)
    
    # Fit in     
    _, out_img, nparms = fit(net=net_fn,
                           num_channels=num_channels,
                           img_shape=x_real.data.shape,
                           num_measurements=hparams.num_measurements,
                           y_feed=y_real,
                           A_feed=A,
                           reg_noise_std=hparams.rn,
                           reg_noise_decayevery=hparams.rnd,
                           LR=0.005,
                           num_iter=hparams.numit,
                           find_best=True,
                           verbose=True)
    out_img = out_img[0]
    
    # Compute and print measurement and l2 loss
    measurement_losses = get_measurement_loss(out_img, A, y_real)
    l2_losses = get_l2_loss(out_img, x_real)
    print ('Final measurement loss is {}'.format(measurement_losses))
    print ('Final representation loss is {}'.format(l2_losses))



if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Input
    PARSER.add_argument('--path', type=str, default='./image/celebA/', help='path stroing the images')
    PARSER.add_argument('--img_name', type=str, default='182659.jpg', help='image to use')
    
    # Measurement type specific hparams
    PARSER.add_argument('--type_measurements', type=str, default='identity', help='measurement type')
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')

    # Deep decoder 
    PARSER.add_argument('--k', type=int, default=128, help='number of channel dimension')
    PARSER.add_argument('--num_channel', type=int, default=6, help='number of upsample channles')
    
    # "Training"
    PARSER.add_argument('--rn', type=float, default=0.005, help='reg_noise_std')
    PARSER.add_argument('--rnd', type=int, default=500, help='reg_noise_decayevery')
    PARSER.add_argument('--numit', type=int, default=2000, help='number of iterations')
   
    HPARAMS = PARSER.parse_args()
    
    main(HPARAMS)

