"""Compressed sensing main script"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #tunable
import numpy as np
import tensorflow as tf
#print('Tensorflow version', tf.__version__) #if your tf version is above 2.0, use next tow lines of code instead 
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from functools import partial
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
from argparse import ArgumentParser
import math
import pandas as pd 
from scipy.fftpack import fft, ifft
import random


# == basic decoder component == #
def _pad(x, kernel_size, pad):
    to_pad = int((kernel_size - 1) / 2)
    if to_pad > 0 and pad == 'reflection':
        return tf.pad(x, ((0, 0), (to_pad, to_pad), (to_pad, to_pad), (0, 0)), mode='REFLECT')
    else:
        return x

def _upsample(i, x, upsample_mode, image_mode, factor=None, layer=None, input_size=128): 
    if upsample_mode == 'none':
        return x
    
    else: 
        if factor:
            w = tf.shape(x)[1]
            h = tf.shape(x)[2]
            w_new = tf.cast(tf.cast(w, tf.float32)*tf.cast(factor, tf.float32), tf.int32)
            h_new = tf.cast(tf.cast(h, tf.float32)*tf.cast(factor, tf.float32), tf.int32)
            if image_mode == '1D':
                new_shape = tf.stack([w_new, h], axis=0)
            elif image_mode == '2D' or '3D':
                new_shape = tf.stack([w_new, h_new], axis=0)
            x = tf.image.resize_images(x, new_shape, align_corners=True, method=getattr(tf.image.ResizeMethod, upsample_mode.upper()))
            return x
        if layer:
            h = tf.shape(x)[2]
            #print('9/11 check alpha: {} i-th layer size: {}'.format(pow(4096/input_size, 1/layer), round(input_size * pow(4096/input_size, i/layer))))
            w_new = tf.cast(round(input_size * pow(4096/input_size, i/layer)), tf.int32)
            h_new = tf.cast(round(input_size * pow(4096/input_size, i/layer)), tf.int32)
            if image_mode == '1D':
                new_shape = tf.stack([w_new, h], axis=0)
            elif image_mode == '2D' or '3D':
                new_shape = tf.stack([w_new, h_new], axis=0)
            x = tf.image.resize_images(x, new_shape, align_corners=True, method=getattr(tf.image.ResizeMethod, upsample_mode.upper()))
            return x
                                   
def _bn(x, bn_affine):
    return tf.layers.batch_normalization(x, trainable=bn_affine, momentum=0.9, epsilon=1e-5, training=True)      


# == decoder variants 0 == #
def decodernw(inputs,
              num_output_channels=3, 
              channels_times_layers=[128] * 6,
              filter_size_up=1,
              need_sigmoid=True, 
              pad='reflection',
              upsample_mode='bilinear',
              image_mode=None,
              act_fun='relu', # tf.nn.relu, tf.nn.leaky_relu, tf.nn.sigmoid
              bn_affine=True,
              factor=None,
              input_size=128
              ):
    """Deep Decoder.
       Takes as inputs a 4D Tensor (batch, width, height, channels)""" #which is the fixed noise with channels = 1
    ## Configure
    n_scales = len(channels_times_layers) #equals to 6
    if act_fun == 'relu':
        activation = tf.nn.relu
    elif act_fun == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif act_fun == 'sigmoid':
        activation = tf.nn.sigmoid
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up] * n_scales #[1, 1 ,...] with shape of [1,6]                                               
        
    ## Deep Decoder
    # B_0
    net = inputs 
    
    # B_1 to B_d
    for i, (num_channels, kernel_size) in enumerate(zip(channels_times_layers, filter_size_up)): 
        # Conv        
        net = tf.layers.conv2d(net, num_channels, kernel_size=kernel_size, strides=1, activation=None, padding='valid', use_bias=False)
        
        # Upsample 
        net = _upsample(i+1, net, upsample_mode, image_mode, factor, len(channels_times_layers), input_size)

        # Activation + Batch Norm           
        net = activation(net)
        net = _bn(net, bn_affine)
 
                
    # output x
    kernel_size = 1
    net = tf.layers.conv2d(net, num_output_channels, kernel_size, strides=1, activation=None, padding='valid', use_bias=False)
    if need_sigmoid:
        net = tf.nn.sigmoid(net)    
    return net #Outputs a 4D Tensor (batch, width*2^6, height*2^6, channels=3) 


# == decoder variants 1&3 == #
def kernel_build(a, iter):
    output = np.zeros((iter, iter, a.shape[0], a.shape[1]))
    for i in range(iter):
        for j in range(iter):
            if i == j:
                output[i,j] = a
    final = output.transpose((2,3,0,1)) #transpose
    return final

def fix_kernel(size, dim, c=2):
    if dim == 2:
        # distance
        def dis(x,y,size):
            center = (size-1)/2
            distance = math.sqrt((center-x)**2 + (center-y)**2)
            return distance
        # matrix
        a = np.ones((size,size))
        for i in range(size):
            for j in range(size):
                a[i][j] = math.exp(-1 * c * dis(i,j,size))        
    elif dim == 1:
        # distance
        def dis(x,size):
            center = (size-1)/2
            distance = math.sqrt((center-x)**2)
            return distance    
        # matrix
        a = np.ones((size,1))
        for i in range(size):
                a[i,:] = math.exp(-1 * c * dis(i,size))        
    # norm
    sum = a.sum()
    a = a/sum
    return np.array(a) 

def kernel_build_assistant(filter_size):
    mtx_1D = fix_kernel(filter_size, 1)
    mtx_2D = fix_kernel(filter_size, 2)
    return mtx_1D, mtx_2D
    
def fixed_decodernw(inputs,
                    num_output_channels=3, 
                    channels_times_layers=[128] * 6,
                    filter_size_up=1,
                    filter_size=3,
                    need_sigmoid=True, 
                    pad='reflection',
                    upsample_mode='bilinear',
                    image_mode=None,
                    act_fun='relu',  # tf.nn.leaky_relu 
                    bn_affine=True,
                    factor=None,
                    input_size=128
                    ):
    """Deep Decoder.
       Takes as inputs a 4D Tensor (batch, width, height, channels)""" #which is the fixed noise with channels = 1
    ## Configure
    n_scales = len(channels_times_layers) #equals to 6
    if act_fun == 'relu':
        activation = tf.nn.relu
    elif act_fun == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif act_fun == 'sigmoid':
        activation = tf.nn.sigmoid
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up = [filter_size_up] * n_scales #[1, 1 ,...] with shape of [1,6]                                               
        
    ## Deep Decoder
    # B_0
    net = inputs
    
    # Fixed kernel
    mtx_1D, mtx_2D = kernel_build_assistant(filter_size)    
    
    if image_mode == '1D':
        filter_np = kernel_build(mtx_1D, channels_times_layers[0])
    elif image_mode == '2D' or '3D':
        filter_np = kernel_build(mtx_2D, channels_times_layers[0])
    filter_cons = tf.constant(filter_np.astype(np.float32))
    filter = tf.Variable(filter_cons, name='fixed_kernel', trainable=False)
    
    # B_1 to B_d 
    for i, (num_channels, kernel_size) in enumerate(zip(channels_times_layers, filter_size_up)):       

        # Conv
        net = tf.nn.conv2d(net, filter, strides=(1,1,1,1), padding='SAME')
        net = tf.layers.conv2d(net, num_channels, kernel_size=kernel_size, strides=1, activation=None, padding='valid', use_bias=False)

        # Upsample
        net = _upsample(i+1, net, upsample_mode, image_mode, factor, len(channels_times_layers), input_size)

        # Batch Norm + activation          
        net = activation(net)
        net = _bn(net, bn_affine) 
                  
    # X output
    kernel_size = 1
    net = tf.layers.conv2d(net, num_output_channels, kernel_size, strides=1, 
                           activation=None, padding='valid', use_bias=False)
    if need_sigmoid:
        net = tf.nn.sigmoid(net)    
    return net #Outputs a 4D Tensor (batch, width*2^6, height*2^6, channels=3)


# == decoder variants 2&4 == #
def deconv_decoder(
        inputs,
        num_output_channels=3, 
        channels_times_layers=[128]*6, 
        filter_size=4,
        image_mode=None,
        pad ='reflection', 
        act_fun='relu', 
        bn_affine=True,
        stride=1,
        need_sigmoid=True, 
        upsample_mode='bilinear',
        factor=None,
        input_size=128
        ):
    
    ## Configure
    n_scales = len(channels_times_layers)
    if act_fun == 'relu':
        activation = tf.nn.relu
    elif act_fun == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif act_fun == 'sigmoid':
        activation = tf.nn.sigmoid
    
    ## Deep decoder 
    # B_0
    net = inputs
    
    # B_1 to B_d
    for i in range(len(channels_times_layers)):
        # TransConv
        if image_mode == '1D':
            net = tf.layers.conv2d_transpose(net, channels_times_layers[i], kernel_size=(filter_size,1), strides=(stride,1), activation=None, padding='same', use_bias=False)
        elif image_mode == '2D' or '3D':
            net = tf.layers.conv2d_transpose(net, channels_times_layers[i], kernel_size=filter_size, strides=stride, activation=None, padding='same', use_bias=False)
        
        # Upsample
        net = _upsample(i+1, net, upsample_mode, image_mode, factor, len(channels_times_layers), input_size)

        # Batch Norm + activation         
        net = activation(net)
        net = _bn(net, bn_affine)            
        
    # X output
    kernel_size = 1
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
        #print('Variable Name: {}, Shape: {}'.format(variable.name, variable.get_shape())) 
        for dim in variable.get_shape():
            params *= dim.value
            #print('Current Dim: {}, Total Params: {}'.format(dim, params))
        total_parameters += params
        #print('Total params for this variable: {}, Total params acumilated: {}'.format(params, total_parameters))
        #print('\t')
    return total_parameters

def save_img(img, path, img_name, name, image_mode, decoder_type, filter_size, upsample_mode, num_channels_real, num_layers, input_size, mask_info='', act_function='relu'):
    img = img[0]
    if image_mode == '1D':
        name = img_name[:-4] + '_' + name + '_' + decoder_type + '_' + str(upsample_mode) +  '_mask_info_' + mask_info + '_filter_size_' + str(filter_size) + '_channels_' + str(num_channels_real) + '_layers_' + str(num_layers) + '_input_size_' + str(input_size) + '_act_' + act_function +'.npy'
        img = img[:,:,0]
        img = img[:,0]
        print('saving shape {}'.format(img.shape))
        #img.tofile(os.path.join(path, name))
        np.save(os.path.join(path, name), img)
    elif image_mode == '2D':
        img = img[:, :, 0]    
        name = img_name[:-4] + '_' + name + '_' + decoder_type + '_' + str(upsample_mode) +  '_mask_info_' + mask_info + '_filter_size_' + str(filter_size) + '_channels_' + str(num_channels_real) + '_layers_' + str(num_layers) + '_input_size_' + str(input_size) + '_act_' + act_function +'.npy'
        print('saving shape {}'.format(img.shape))
        #img.tofile(os.path.join(path, name))
        np.save(os.path.join(path, name), img)
    else:
        img = img * 255.
        name = img_name[:-4] + '_' + name + '_' + decoder_type + '_' + str(upsample_mode) +  '_mask_info_' + mask_info + '_filter_size_' +  str(filter_size) + '_channels_' + str(num_channels_real) + '_layers_' + str(num_layers) + '_input_size_' + str(input_size) + '_act_' + act_function +'.jpg'
        imsave(os.path.join(path, name), img.astype(np.uint8))

def fit(net,
        upsample_factor,
        channels_times_layers,
        img_shape,
        image_mode,
        decoder_type,
        upsample_mode,
        filter_size,
        img_name, 
        type_measurements,
        num_measurements,
        num_channels_real,
        num_layers,
        act_function,
        y_feed,
        A_feed,
        mask_info1,
        mask_info2,
        mask_feed = None, 
        lr_decay_epoch=0,
        lr_decay_rate=0.65,
        LR=0.01,
        OPTIMIZER='adam',
        num_iter=5000,
        find_best=False,
        verbose=False,
        device='gpu:0',
        input_size = 128,
        random_vector = None, 
        selection_mask = None,
        save = True
       ):
    """Fit a model.
    
        Args: 
        net: the generative model
        channels_times_layers: Number of upsample channels #e.g.[k, k ,...] with shape of [1,6]
        img_shape: original real image shape, a 4D tensor, e.g. [1,64,64,3] 
        type_measurements, num_measurements: the type and number of measurements 
        y_feed, A_feed: real oberservation y and measurment matrix A
        LR, lr_decay_epoch, lr_decay_rate: parameters of learning rate 
        device: device name 
        
    """
    
    with tf.Graph().as_default():
        # Global step
        global_step = tf.train.get_or_create_global_step()
            
        with tf.device('/%s' % device):  
            # Set up palceholders
            if mask_feed is None:
                n_input = img_shape[1]*img_shape[2]*img_shape[3]
                if type_measurements == 'random':
                    A = tf.placeholder(tf.float32, shape=(n_input, num_measurements), name='A') #e.g.[img_wid*img_high*3, 200]
                    y = tf.placeholder(tf.float32, shape=(1, num_measurements), name='y') #e.g.[1, 200]
                elif type_measurements == 'identity':
                    A = tf.placeholder(tf.float32, shape=(n_input, n_input), name='A') #e.g.[img_wid*img_high*3, img_wid*img_high*3]
                    y = tf.placeholder(tf.float32, shape=(1, n_input), name='y') #e.g.[1, img_wid*img_high*3]
                elif type_measurements == 'circulant':
                    y = tf.placeholder(tf.float32, shape=(1, n_input), name='y')#e.g.[1, img_wid*img_high*3]
            else:
                y = tf.placeholder(tf.float32, shape=(1, img_shape[1], img_shape[2], img_shape[3]), name='y')
            
            # Define input uniform noise 
            if upsample_mode == 'bilinear':
                ## -- fix output size only --##
                #totalupsample = upsample_factor**len(num_layers) #e.g. 2^6, 1.5^3
                #width = int(img_shape[1] / totalupsample)
                #if image_mode == '1D':
                #    height = int(img_shape[2])
                #elif image_mode == '2D' or '3D':    
                #    height = int(img_shape[2] / totalupsample)
                
                ## -- fix input size and output size--: ##
                width = input_size
                if image_mode == '1D':
                    height = int(img_shape[2])
                elif image_mode == '2D' or '3D':    
                    height = input_size
                #print('9/11 input noise check, width: {} height:{}'.format(width, height))
            elif upsample_mode == 'none':
                width = int(img_shape[1])
                height = int(img_shape[2])
            z = tf.constant(np.random.uniform(size=[1, width, height, channels_times_layers[0]]).astype(np.float32) * 1. / 10)
            z = tf.Variable(z, name='z', trainable=False)
            
            # Deep decoder prior 
            feed_forward = tf.make_template("DeepDecoder", net) #feed_forward takes a 4D Tensor (batch, width, height, channels) as input and outputs a 4D Tensor (batch, width*2^6, height*2^6, channels=3)
            x = feed_forward(z) #net_output with shape [1, img_wid, img_high, 3]               
            
            # Inverse problem 
            def circulant_tf(signal_vector, random_vector, signal_size, selection_mask):  
                signal_vector = tf.cast(signal_vector, dtype=tf.complex64, name='circulant_real2complex')
                t = tf.convert_to_tensor(random_vector, dtype=tf.complex64)
                #step 1: F^{-1} @ x
                r1 = tf.signal.ifft(signal_vector, name='circulant_step1_ifft')               
                #step 2: Diag() @ F^{-1} @ x
                Ft = tf.signal.fft(t)
                r2 = tf.multiply(r1, Ft, name='circulant_step2_diag')                
                #step 3: F @ Diag() @ F^{-1} @ x
                compressive = tf.signal.fft(r2, name='circulant_step3_fft')
                float_compressive = tf.cast(compressive, tf.float32, name='circulant_complex2real')               
                #step 4: R_{omega} @ C_{t}
                select_compressive = tf.multiply(float_compressive, selection_mask, name='circulant_step4_A')            
                return select_compressive
            
            if mask_feed is None:
                # Compressed sensing / Denoising     
                if type_measurements == 'circulant':
                    # Flip
                    a = []
                    for i in range(tf.reshape(x, [1,-1]).shape.as_list()[-1]):
                        if random.randint(1, 10000) <= int(10001/2):
                            a.append(1)
                        else:
                            a.append(-1)
                    arr = np.array(a)
                    flip = tf.convert_to_tensor(arr, dtype=tf.float32)
                    x_circulant =  tf.reshape(x, [1,-1]) * flip 
                    # Circulant
                    y_hat = circulant_tf(x_circulant, random_vector, n_input, selection_mask)
                else:
                    y_hat = tf.matmul(tf.reshape(x, [1,-1]), A, name='y_hat')
            else:
                # Inpainting 
                y_hat = x * mask_feed
                
            # Define loss  
            mse = tf.losses.mean_squared_error
            loss = mse(y, y_hat)            

            # Define learning rate 
            if lr_decay_epoch > 0:
                LR = tf.train.exponential_decay(LR, global_step, lr_decay_epoch, lr_decay_rate, staircase=True)

            # Define optimizer 
            if OPTIMIZER == 'adam':
                #print("optimize with adam", LR)
                optimizer = tf.train.AdamOptimizer(LR)
            elif OPTIMIZER == 'LBFGS':
                raise NotImplementedError('LBFGS Optimizer')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)    

        with tf.Session() as sess:
            # Init            
            mse = [0.] * num_iter
            sess.run(tf.global_variables_initializer())    
                        
            # Initial deep decoder output
            if find_best:
                if not os.path.exists('log'):
                    os.makedirs('log/')
                if not os.path.exists('result'):
                    os.makedirs('result/')
                saver = tf.train.Saver(max_to_keep=1)
                #saver.save(sess, os.path.join('log/', 'model.ckpt'), global_step=0)
                best_mse = 1000000.0
                best_img = sess.run(x)
                #save_img(best_img, 'result/', img_name, '0', image_mode, decoder_type, filter_size, upsample_mode) 
            
            # Feed dict
            if mask_feed is None:
                if type_measurements == 'circulant':
                    feed_dict = {y: y_feed}
                else:
                    feed_dict = {A: A_feed, y: y_feed}
            else:
                feed_dict = {y: y_feed}
                
            # Desired noised/masked output
            #y_recov = sess.run(y, feed_dict=feed_dict) 
            #y_name = 'y_recov_ini' + '_' + decoder_type + '_' + str(filter_size) + '.npy'
            #imsave(os.path.join('result/', y_name), y_recov.astype(np.uint8))
            
            # Optimize
            num_params = get_num_params()
            sess.graph.finalize()
            #print('\x1b[37mFinal graph size: %.2f MB\x1b[0m' % (tf.get_default_graph().as_graph_def().ByteSize() / 10e6))

            for i in range(num_iter):
                loss_, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                #psnr = 10 * np.log10(1 * 1 / loss_) #PSNR
                mse[i] = loss_
       
                # Display
                #if i > 0 and i % 100 == 0:
                #    print ('\r[Iteration %05d] loss=%.9f' % (i, loss_), end='')
                
                # Best net
                if find_best and best_mse > 1.005 * loss_:
                    best_mse = loss_
                    #best_psnr = 10 * np.log10(1 * 1 / best_mse)
                    best_img = sess.run(x)
                    #saver.save(sess, os.path.join('log/', 'model.ckpt'), global_step=i + 1)
                         
            # Return final image or best found so far if `find_best`
            if find_best:
                out_img = best_img
                if image_mode == '1D':
                    mask_info = mask_info1[8:-4]
                elif image_mode == '2D' or '3D':
                    mask_info = mask_info2[8:-4]
                if save:
                    save_img(out_img, 'result/', img_name, '{}'.format(i + 1), image_mode, decoder_type, filter_size, upsample_mode, num_channels_real, num_layers, input_size, mask_info, act_function)
                #print('Best MSE (wrt noisy) {}: {}: {}: {}: {}: {}: {}: {}: {}'.format(num_channels_real, num_layers, img_name, mask_info, decoder_type, filter_size, upsample_mode, upsample_factor, best_mse))
            else:
                out_img = sess.run(x)
                if image_mode == '1D':
                    mask_info = mask_info1[8:-4]
                elif image_mode == '2D' or '3D':
                    mask_info = mask_info2[8:-4]
                if save:
                    save_img(out_img, 'result/', img_name, '{}'.format(i + 1), image_mode, decoder_type, filter_size, upsample_mode, num_channels_real, num_layers, input_size, mask_info, act_function)
                #print('FINAL MSE (wrt noisy) {}: {}: {}: {}: {}: {}: {}: {}: {}'.format(num_channels_real, num_layers, img_name, mask_info, decoder_type, filter_size, upsample_mode, upsample_factor, mse[-1]))
            if verbose:
                return mse, out_img, num_params
            else:
                return mse, out_img
    
    
    
  
# == main part == #
def load_1D(path, img_name, type, flip):
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    # Flip
    if type == 'denoising' and flip == 'circulant':
        for i in range(img.shape[0]):
            if random.randint(1, 10000) <= int(10001/2):
                img[i] = img[i]*-1
    img = img[:, None]
    img = img[:, :, None]
    img = img[None, :, :, :]
    return img

def load_2D(path, img_name): #might need to be fliped later 
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = img[None, :, :, :]
    return img
    
def load_img(path, img_name):
    img_path = os.path.join(path, img_name)
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = img[None, :, :, :]
    img_clean = img / 255.
    return img_clean
     
def get_l2_loss(image1, image2):
    assert image1.shape == image2.shape
    return np.mean((image1 - image2)**2)
    
def load_mask(path, img_name): 
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    if len(img.shape) == 1:
        img = img[:, None]
    img = img[:, :, None]
    img = img[None, :, :, :]
    return img

def create_A_selection(signal_size, measurement_size):
    mask_ = np.zeros((1,signal_size))
    idx_list = np.random.choice(signal_size, measurement_size, replace=False)
    for idx in idx_list:
        mask_[:,idx] = 1.0 
    return mask_

def main(hparams):
    # Get inputs
    if hparams.image_mode == '1D':
        x_real = np.array(load_1D(hparams.path, hparams.img_name, hparams.model_type, hparams.type_measurements)).astype(np.float32)
    elif hparams.image_mode == '2D':
        x_real = np.array(load_2D(hparams.path, hparams.img_name)).astype(np.float32)
    else:
        x_real = np.array(load_img(hparams.path, hparams.img_name)).astype(np.float32)
    
    # Construct measurements/noises for compressed sensing 
    mea_shape = x_real.shape[1] * x_real.shape[2] * x_real.shape[3] #i.e. img_wid*img_wid*3
    random_vector = None #initialization 
    A = None #initialization
    selection_mask = None #initialization
    if hparams.type_measurements == 'random':
        A = np.random.randn(mea_shape, hparams.num_measurements).astype(np.float32)
        noise_shape = [1, hparams.num_measurements]
    elif hparams.type_measurements == 'identity':
        A = np.identity(mea_shape).astype(np.float32)
        noise_shape = [1, mea_shape]
    elif hparams.type_measurements == 'circulant':
        random_vector = np.random.normal(size=mea_shape)
        selection_mask = create_A_selection(mea_shape, hparams.num_measurements)
        noise_shape = [1, mea_shape]   
        
    def circulant_np(signal_vector, random_vector, signal_size, selection_mask):
        #step 1: F^{-1} @ x
        r1 = ifft(signal_vector)
        #step 2: Diag() @ F^{-1} @ x
        Ft = fft(random_vector) 
        r2 = np.multiply(r1, Ft)
        #step 3: F @ Diag() @ F^{-1} @ x
        compressive = fft(r2)
        #step 4:  R_{omega} @ C_{t} @ D){epsilon}
        compressive = compressive.real
        select_compressive = compressive * selection_mask
        return select_compressive
    
    observ_noise = hparams.noise_level * np.random.randn(noise_shape[0], noise_shape[1])
    
    # Construct mask for inpainting
    if hparams.model_type == 'inpainting':
        if hparams.image_mode == '1D':
            mask = load_mask(hparams.path, hparams.mask_name_1D)
        elif image_mode == '2D' or '3D':
            mask = load_mask(hparams.path, hparams.mask_name_2D)
    elif hparams.model_type == 'denoising':
        mask = None 
    
    # Construct observation 
    if hparams.model_type == 'inpainting':
        y_real = x_real * mask #shape [1, img_wid, img_high, channels]
    elif hparams.model_type == 'denoising':
        if hparams.type_measurements == 'circulant':
            y_real = circulant_np(x_real.reshape(1,-1), random_vector, mea_shape, selection_mask) + observ_noise
        else:
            y_real = np.matmul(x_real.reshape(1,-1), A) + observ_noise #noise level with shape [10,100]
    #y_name = hparams.image_mode + '_y_' + hparams.decoder_type + '_' + str(hparams.filter_size) + '_' + hparams.upsample_mode + '.npy'
    #np.save(os.path.join('result/', y_name), y_real[0,:,:,0])
    
    # Define channles*layers 
    channels_times_layers = [hparams.k]*hparams.num_layers   

    # Define activations
    act_function = hparams.activation
    
    # Define decoder network 
    if hparams.decoder_type == 'original': 
        net_fn =  partial(decodernw, num_output_channels=x_real.shape[-1], channels_times_layers=channels_times_layers, image_mode=hparams.image_mode, 
                upsample_mode=hparams.upsample_mode, factor=hparams.upsample_factor, input_size=hparams.input_size, act_fun=act_function)
    elif hparams.decoder_type == 'fixed_deconv':  
        net_fn =  partial(fixed_decodernw, num_output_channels=x_real.shape[-1], channels_times_layers=channels_times_layers, image_mode=hparams.image_mode, 
                upsample_mode=hparams.upsample_mode, filter_size=hparams.filter_size, factor=hparams.upsample_factor, input_size=hparams.input_size, act_fun=act_function)
    elif hparams.decoder_type == 'deconv':  
        net_fn =  partial(deconv_decoder, num_output_channels=x_real.shape[-1], channels_times_layers=channels_times_layers, image_mode=hparams.image_mode, 
                upsample_mode=hparams.upsample_mode, filter_size=hparams.filter_size, factor=hparams.upsample_factor, input_size=hparams.input_size, act_fun=act_function)

    # Fit in
    #print('current number of channels: {}'.format(hparams.k))
    mse, out_img, nparms = fit(net=net_fn,
                           upsample_factor=hparams.upsample_factor,
                           channels_times_layers=channels_times_layers,
                           img_shape=x_real.shape,
                           image_mode=hparams.image_mode,
                           decoder_type=hparams.decoder_type,
                           filter_size=hparams.filter_size,
                           upsample_mode=hparams.upsample_mode,
                           img_name=hparams.img_name,                           
                           type_measurements=hparams.type_measurements,
                           num_measurements=hparams.num_measurements,
                           num_channels_real = hparams.k,
                           num_layers = hparams.num_layers,
                           act_function = act_function,
                           y_feed=y_real,
                           A_feed=A,
                           mask_info1=hparams.mask_name_1D,
                           mask_info2=hparams.mask_name_2D,
                           mask_feed=mask,
                           LR=0.005,
                           num_iter=hparams.numit,
                           find_best=True,
                           verbose=True,
                           input_size=hparams.input_size,
                           random_vector=random_vector,
                           selection_mask=selection_mask,
                           save = True)
    #out_img = out_img[0] #4D tensor to 3D tensor if need to plot 
    
    # Compute and print measurement and l2 loss
    measurement_losses = mse[-1]
    l2_losses = get_l2_loss(out_img, x_real)
    psnr = 10 * np.log10(1 * 1 / l2_losses) #PSNR
    #print ('Final measurement loss is {}'.format(measurement_losses))
    print ('Final number of params is {}'.format(nparms))
    
    if hparams.model_type == 'inpainting':
        if hparams.image_mode == '1D':
            mask_info = hparams.mask_name_1D[8:-4]
        elif image_mode == '2D' or '3D':
            mask_info = hparams.mask_name_2D[8:-4]
        type_mea_info = 'NA'
        num_mea_info = 'NA'
    else:
        mask_info = 'NA'
        type_mea_info = hparams.type_measurements
        num_mea_info = hparams.num_measurements
    
    if hparams.upsample_factor:
        factor_record = hparams.upsample_factor
    else:
        factor_record = round(1 * pow(4096/hparams.input_size, 1/hparams.num_layers), 4)
        
    print ('Final representation PSNR for img_name:{}, model_type:{}, type_mea:{}, num_mea:{}, mask:{}, decoder:{}, \
          filter_size:{}, upsample_mode:{}, #channels:{} #layers:{} up_factor:{} input_size:{} act_func:{} is {}'.format(hparams.img_name, 
            hparams.model_type, type_mea_info, num_mea_info, mask_info, 
            hparams.decoder_type, hparams.filter_size, hparams.upsample_mode, hparams.k, hparams.num_layers, factor_record, hparams.input_size, hparams.activation, psnr))
    
    # To pd frame 
    if hparams.hyperband_mode == 0:
        pickle_file_path = hparams.pickle_file_path
        if not os.path.exists(pickle_file_path):
            d = {'img_name':[hparams.img_name], 'model_type':[hparams.model_type], 'type_mea':[type_mea_info], 'num_mea':[num_mea_info], 'mask_info':[mask_info], 
                'decoder_type':[hparams.decoder_type], 'filter_size':[hparams.filter_size], 'upsample_mode':[hparams.upsample_mode], 
                'channels':[hparams.k], 'layers':[hparams.num_layers], 'up_factor':[factor_record], 'input_size':[hparams.input_size], 'act_func':[hparams.activation], 'psnr':[psnr]}
            df = pd.DataFrame(data=d)
            df.to_pickle(pickle_file_path)
        else:
            d = {'img_name':hparams.img_name, 'model_type':hparams.model_type, 'type_mea':type_mea_info, 'num_mea':num_mea_info, 'mask_info':mask_info, 
                'decoder_type':hparams.decoder_type, 'filter_size':hparams.filter_size, 'upsample_mode':hparams.upsample_mode, 
                'channels':hparams.k, 'layers':hparams.num_layers, 'up_factor':factor_record, 'input_size':hparams.input_size, 'act_func':hparams.activation, 'psnr':psnr}
            df = pd.read_pickle(pickle_file_path)
            df = df.append(d, ignore_index=True)
            df.to_pickle(pickle_file_path)
    
    # End
    print('END')
    print('\t')




if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images') 
    PARSER.add_argument('--path', type=str, default='image/Gaussian signal', help='path stroing the images')
    PARSER.add_argument('--noise_level', type=float, default=0.00, help='std dev of noise') 
    PARSER.add_argument('--img_name', type=str, default='1D_rbf_2.npy', help='image to use') 
    PARSER.add_argument('--model_type', type=str, default='inpainting', help='inverse problem model type') 
    PARSER.add_argument('--type_measurements', type=str, default='circulant', help='measurement type') 
    PARSER.add_argument('--num_measurements', type=int, default=500, help='number of gaussian measurements') 
    PARSER.add_argument('--mask_name_1D', type=str, default='mask/1D_rbf_3.0_1.npy', help='mask to use') 
    PARSER.add_argument('--mask_name_2D', type=str, default='mask/2D_rbf_3.0_1.npy', help='mask to use') 
    PARSER.add_argument('--pickle_file_path', type=str, default='result/cs_result.pkl') 
    PARSER.add_argument('--hyperband_mode', type=int, default=0) 

    # Deep decoder 
    PARSER.add_argument('--k', type=int, default=256, help='number of channel dimension') 
    PARSER.add_argument('--num_layers', type=int, default=6, help='number of upsample layers')
    PARSER.add_argument('--decoder_type', type=str, default='fixed_deconv', help='decoder type') 
    PARSER.add_argument('--upsample_mode', type=str, default='bilinear', help='upsample type') 
    PARSER.add_argument('--filter_size', type=int, default=8, help='upsample type') 
    PARSER.add_argument('--upsample_factor', type=float, default=None, help='upsample factor') 
    PARSER.add_argument('--input_size', type=int, default=128, help='input_size') 
    PARSER.add_argument('--activation', type=str, default='relu', help='activation type')
    
    # "Training"
    PARSER.add_argument('--rn', type=float, default=0, help='reg_noise_std')
    PARSER.add_argument('--rnd', type=int, default=500, help='reg_noise_decayevery')
    PARSER.add_argument('--numit', type=int, default=10000, help='number of iterations')
   
    HPARAMS = PARSER.parse_args()
    
    main(HPARAMS)

