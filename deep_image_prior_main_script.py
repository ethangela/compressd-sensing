"""Compressed sensing main script"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#print('Tensorflow version', tf.__version__)
from functools import partial
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
#from skimage.color import rgb2gray, rgba2rgb
#from matplotlib import pyplot as plt
from argparse import ArgumentParser
import math
import pandas as pd 
#from scipy.linalg import dft
from scipy.fftpack import fft, ifft
#from scipy.sparse import diags
#from scipy import linalg
import random




## -- deep image prior -- ##
def down_layer(layer, image_mode): #size-down 2, chn -> 128
    if image_mode == '1D':
        #1st down
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(3,1), strides=(2,1), activation=None, padding='same') 
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
        #2nd down 
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(3,1), strides=(1,1), activation=None, padding='same')
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
    else:
        #1st down
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(3,3), strides=(2,2), activation=None, padding='same') 
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
        #2nd down 
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(3,3), strides=(1,1), activation=None, padding='same')
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True)  
    #output
    return layer

def up_layer(layer, image_mode):#size-up 2, chn -> 128
    layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
    if image_mode == '1D':
        #1st up
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(3,1), strides=(1,1), activation=None, padding='same') 
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
        #2nd up
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(1,1), strides=(1,1), activation=None, padding='same') 
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
        #size up
        height, width = layer.get_shape()[1:3] #[256,256]
        layer = tf.image.resize(images = layer, size = [height*2, width]) #[512,512]
    else:
        #1st up
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(3,3), strides=(1,1), activation=None, padding='same') 
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
        #2nd up
        layer = tf.layers.conv2d(layer, filters=128, kernel_size=(1,1), strides=(1,1), activation=None, padding='same') 
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.batch_normalization(layer, trainable=True, training=True) 
        #size up
        height, width = layer.get_shape()[1:3] #[256,256]
        layer = tf.image.resize(images = layer, size = [height*2, width*2]) #[512,512]
    #output
    return layer

def skip(layer): #chn -> 4
    layer = tf.layers.conv2d(layer, filters=4, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False) 
    layer = tf.nn.leaky_relu(layer)
    conv_out = tf.layers.batch_normalization(layer, trainable=True, training=True)
    #output
    return conv_out

def gkern(kernlen=5, nsig=3): 
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return (tf.convert_to_tensor(kernel, dtype=tf.float32))

def gblur(layer):  # Apply the gaussian kernel to each channel to give the image a gaussian blur.
    gaus_filter = tf.expand_dims(tf.expand_dims(gkern(), axis=-1), axis=-1)
    return tf.nn.depthwise_conv2d(layer, gaus_filter, strides=[1,1,1,1], padding='SAME')

def deep_image_prior(inputs, down_layer_count, up_layer_count, num_output_channels, image_mode):
    # Inputs
    out = inputs

    # Connect up all the downsampling layers.
    skips = []
    for i in range(down_layer_count): #size: dim -> dim/(2^count), chn: 32 -> 128
        out = down_layer(out, image_mode) 
        skips.append(skip(out)) # Keep a list of the skip layers, so they can be connected to the upsampling layers
    print("Shape after downsample: " + str(out.get_shape()))

    # Connect up the upsampling layers, from smallest to largest.
    skips.reverse()
    for i in range(up_layer_count): #size: dim/(2^count) -> dim, chn: 128 -> 128
        if i == 0:
            # As specified in the paper, the first upsampling layers is connected to the last downsampling layer through a skip layer.
            out = up_layer(skip(out), image_mode)
        else:
            # The output of the rest of the skip layers is concatenated onto
            # the input of each upsampling layer.
            # Note: It's not clear from the paper if concat is the right operator
            # but nothing else makes sense for the shape of the tensors.
            out = up_layer(tf.concat([out, skips[i]], axis=3), image_mode)    
    print("Shape after upsample: " + str(out.get_shape()))

    # Restore original image dimensions and channels 
    out = tf.layers.conv2d(out, filters=num_output_channels, kernel_size=(1,1), strides=(1,1), padding='same', activation=tf.nn.sigmoid)#[1,dim,dim,num_output_channels]
    print("Output shape: " + str(out.get_shape())) 
    return out




## -- fit part -- ##
def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables(scope="DeepImagePrior"):
        params = 1
        #print('Variable Name: {}, Shape: {}'.format(variable.name, variable.get_shape())) 
        for dim in variable.get_shape():
            params *= dim.value
            #print('Current Dim: {}, Total Params: {}'.format(dim, params))
        total_parameters += params
        #print('Total params for this variable: {}, Total params acumilated: {}'.format(params, total_parameters))
        #print('\t')
    return total_parameters

def fit(net,
        img_shape,
        img_name, 
        image_mode,
        type_measurements,
        num_measurements,
        y_feed,
        A_feed,
        mask_info1,
        ini_channel = 32,
        mask_feed = None, 
        lr_decay_epoch=0,
        lr_decay_rate=0.65,
        LR=0.01,
        OPTIMIZER='adam',
        num_iter=5000,
        find_best=False,
        verbose=False,
        random_vector = None, 
        selection_mask = None,
        save = False,
        random_array = None):
    
    with tf.Graph().as_default():
        # Global step
        global_step = tf.train.get_or_create_global_step()
        
        # Set up palceholders
        n_input = img_shape[1]*img_shape[2]*img_shape[3]
        width = int(img_shape[1])
        height = int(img_shape[2])
        if mask_feed is None:
            if type_measurements == 'random': #compressed sensing with random matirx 
                A  = tf.placeholder(tf.float32, shape=(n_input, num_measurements), name='A') #e.g.[img_wid*img_high*3, 200]
                y = tf.placeholder(tf.float32, shape=(1, num_measurements), name='y') #e.g.[1, 200]
                #rand = tf.placeholder(tf.float32, shape=(1, width, height, ini_channel), name='random_noise') #e.g.[1,img_wid,img_high,32] 
            elif type_measurements == 'identity': #denosing 
                if image_mode != '3D':
                    A = tf.placeholder(tf.float32, shape=(n_input, n_input), name='A') #e.g.[img_wid*img_high*3, img_wid*img_high*3] ########!!!!!!#####!!!!!!!
                y = tf.placeholder(tf.float32, shape=(1, n_input), name='y') #e.g.[1, img_wid*img_high*3]
                #rand = tf.placeholder(tf.float32, shape=(1, width, height, ini_channel), name='random_noise') #e.g.[1,img_wid,img_high,32] 
            elif type_measurements == 'circulant': #compressed sensing with circulant matirx 
                y = tf.placeholder(tf.float32, shape=(1, n_input), name='y')#e.g.[1, img_wid*img_high*3]
                #rand = tf.placeholder(tf.float32, shape=(1, width, height, ini_channel), name='random_noise') #e.g.[1,img_wid,img_high,32] 
        else: #inpainting
            y = tf.placeholder(tf.float32, shape=(1, img_shape[1], img_shape[2], img_shape[3]), name='y')#e.g.[1, img_wid, img_high, 3]
            #rand = tf.placeholder(tf.float32, shape=(1, width, height, ini_channel), name='random_noise') #e.g.[1,img_wid,img_high,32] 
        
        # Define input uniform noise
        #rand = np.random.uniform(0, 1.0/30.0, size=(1, width, height, ini_channel)).astype(np.float32)
        out = tf.constant(np.random.uniform(size=(1, width, height, ini_channel)).astype(np.float32) * 1. / 10) #+ rand  #[1,4096,1,32] 
        out = tf.Variable(out, name='input_noise', trainable=False)
        
        # Deep image prior 
        feed_forward = tf.make_template("DeepImagePrior", net) #feed_forward takes a 4D Tensor (batch, width, height, channels) as input and outputs a 4D Tensor (batch, width*2^6, height*2^6, channels=3)
        x = feed_forward(out) #e.g. net_output with shape [1, img_wid, img_high, 3]               
        
        # Inverse problem settings
        def circulant_tf(signal_vector, random_vector_m, selection_mask_m):  
            signal_vector = tf.cast(signal_vector, dtype=tf.complex64, name='circulant_real2complex')
            t = tf.convert_to_tensor(random_vector_m, dtype=tf.complex64)
            #step 1: F^{-1} @ x
            r1 = tf.signal.ifft(signal_vector, name='circulant_step1_ifft')               
            #step 2: Diag() @ F^{-1} @ x
            Ft = tf.signal.fft(t)
            r2 = tf.multiply(r1, Ft, name='circulant_step2_diag')                
            #step 3: F @ Diag() @ F^{-1} @ x
            compressive = tf.signal.fft(r2, name='circulant_step3_fft')
            float_compressive = tf.cast(compressive, tf.float32, name='circulant_complex2real')               
            #step 4: R_{omega} @ C_{t}
            select_compressive = tf.multiply(float_compressive, selection_mask_m, name='circulant_step4_A')            
            return select_compressive
        
        if mask_feed is None: # Compressed sensing & Denoising      
            if type_measurements == 'circulant': # Compressed sensing with Circulant matrix 
                flip = tf.convert_to_tensor(random_array, dtype=tf.float32) # flip
                x_circulant =  tf.reshape(x, [1,-1]) * flip 
                y_hat = circulant_tf(x_circulant, random_vector, selection_mask) 
            else: # Compressed sensing with Random matrix & Denoising 
                if image_mode != '3D':
                    y_hat = tf.matmul(tf.reshape(x, [1,-1]), A) ########!!!!!!#####!!!!!!!
                else:
                    y_hat = tf.reshape(x, [1,-1])
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

        # Set up gpu
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.85 
        config.log_device_placement= True
        
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
                if type_measurements == 'circulant':#compressed sensing
                    feed_dict = {y: y_feed}
                elif type_measurements == 'identity':
                    if image_mode != '3D':
                        feed_dict = {A: A_feed, y: y_feed}  ########!!!!!!#####!!!!!!!
                    else:
                        feed_dict = {y: y_feed}
            else:#inpainting
                feed_dict = {y: y_feed}
                            
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
                #mask_info = mask_info1[8:-4]
                # if save:
                #     save_img(out_img, 'result/', img_name, '{}'.format(i + 1), image_mode, decoder_type, filter_size, upsample_mode, num_channels_real, num_layers, input_size, mask_info, act_function)
                #print('Best MSE (wrt noisy) {}: {}: {}: {}: {}: {}: {}: {}: {}'.format(num_channels_real, num_layers, img_name, mask_info, decoder_type, filter_size, upsample_mode, upsample_factor, best_mse))
            else:
                out_img = sess.run(x)
                #mask_info = mask_info1[8:-4]
                # if save:
                #     save_img(out_img, 'result/', img_name, '{}'.format(i + 1), image_mode, decoder_type, filter_size, upsample_mode, num_channels_real, num_layers, input_size, mask_info, act_function)
                #print('FINAL MSE (wrt noisy) {}: {}: {}: {}: {}: {}: {}: {}: {}'.format(num_channels_real, num_layers, img_name, mask_info, decoder_type, filter_size, upsample_mode, upsample_factor, mse[-1]))
            if verbose:
                return mse, out_img, num_params
            else:
                return mse, out_img




# == main part == #
def random_flip(value):
    a = []
    for i in range(value):
        if random.randint(1, 10000) <= int(10001/2):
            a.append(1)
        else:
            a.append(-1)
    arr = np.array(a)
    return arr 

def load_1D(path, img_name):
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    img = img[:, None]
    img = img[:, :, None]
    img = img[None, :, :, :]
    return img
     
def load_2D(path, img_name):
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = img[None, :, :, :]
    return img
    
def load_img(path, img_name):
    img_path = os.path.join(path, img_name)
    img = imread(img_path)
    img = np.transpose(img, (1, 0, 2))
    img = img[25:153,45:173,:] ####!!!!!!!!!!!
    #img = np.pad(img, ((23,23), (3,3), (0,0)), 'constant', constant_values=0) #[178,218,3] -> [224,224,3] ####!!!!!!!!!!!
    img = img[None, :, :, :]
    img_clean = img / 255.
    return img_clean

def get_l2_loss(image1, image2, image_mode):
    assert image1.shape == image2.shape
    #if image_mode == '3D':
    #    image1, image2 = image1[0,23:201,3:221,:], image2[0,23:201,3:221,:] ####!!!!!!!!!!!
    return np.mean((image1 - image2)**2)
    
def load_mask(path, img_name): 
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    if len(img.shape) == 1:
        img = img[:, None]
    #elif len(img.shape) == 2:
    #    img = np.pad(img, ((23,23), (3,3)), 'constant', constant_values=0) #[178,218] -> [224,224] ####!!!!!!!!!!!
    img = img[:, :, None]
    img = img[None, :, :, :]
    return img

def create_A_selection(signal_size, measurement_size):
    mask_ = np.zeros((1,signal_size))
    idx_list = np.random.choice(signal_size, measurement_size, replace=False)
    for idx in idx_list:
        mask_[:,idx] = 1.0 
    return mask_

def save_out_img(img, path, img_name, decoder_type, model_type, num_mea_info, mask_info, noise_level, image_mode): ####!!!!!!!!!!!
    if image_mode == '1D':
        img = img[:,:,:,0]
        img = img[:,:,0]
        img = img[0,:]
        if model_type == 'inpainting':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_mask_' + mask_info +'.npy'
        elif model_type == 'denoising':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_noise_' + noise_level +'.npy'
        elif model_type == 'compressing':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_num_mea_' + num_mea_info +'.npy'
        np.save(os.path.join(path, name), img)
    elif image_mode == '2D':
        img = img[:,:,:,0]
        img = img[0,:,:]
        if model_type == 'inpainting':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_mask_' + mask_info +'.npy'
        elif model_type == 'denoising':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_noise_' + noise_level +'.npy'
        elif model_type == 'compressing':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_num_mea_' + num_mea_info +'.npy'
        np.save(os.path.join(path, name), img)
    elif image_mode == '3D':
        #img = img[0,23:201,3:221,:] ####!!!!!!!!!!!
        img = img * 255.
        img = np.transpose(img, (1, 0, 2))
        if model_type == 'inpainting':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_mask_' + mask_info +'.jpg'
        elif model_type == 'denoising':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_noise_' + noise_level +'.jpg'
        elif model_type == 'compressing':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_num_mea_' + num_mea_info +'.jpg'
        imsave(os.path.join(path, name), img.astype(np.uint8))
    print('saving shape {}'.format(img.shape))
    #img.tofile(os.path.join(path, name))

def main(hparams):
    # Get inputs
    if hparams.image_mode == '1D':
        x_real = load_1D(hparams.path, hparams.img_name)
        x_real = np.array(x_real).astype(np.float32)
    elif hparams.image_mode == '2D':
        x_real = np.array(load_2D(hparams.path, hparams.img_name)).astype(np.float32)
    elif hparams.image_mode == '3D':
        x_real = np.array(load_img(hparams.path, hparams.img_name)).astype(np.float32)

    # Construct measurements/noises for compressed sensing and denoising 
    mea_shape = x_real.shape[1] * x_real.shape[2] * x_real.shape[3] #i.e. img_wid*img_wid*3
    random_vector = None #initialization 
    A = None #initialization
    selection_mask = None #initialization
    if hparams.type_measurements == 'random': #compressed sensing
        A = np.random.randn(mea_shape, hparams.num_measurements).astype(np.float32)
        noise_shape = [1, hparams.num_measurements]
    elif hparams.type_measurements == 'identity': #denoising 
        if hparams.image_mode != '3D':
            A = np.identity(mea_shape).astype(np.float32) ########!!!!!!#####!!!!!!!
        noise_shape = [1, mea_shape]
    elif hparams.type_measurements == 'circulant': #compressed sensing
        random_vector = np.random.normal(size=mea_shape)
        selection_mask = create_A_selection(mea_shape, hparams.num_measurements)
        noise_shape = [1, mea_shape]   
        
    def circulant_np(signal_vector, random_vector_m, selection_mask_m):
        #step 1: F^{-1} @ x
        r1 = ifft(signal_vector)
        #step 2: Diag() @ F^{-1} @ x
        Ft = fft(random_vector_m) 
        r2 = np.multiply(r1, Ft)
        #step 3: F @ Diag() @ F^{-1} @ x
        compressive = fft(r2)
        #step 4:  R_{omega} @ C_{t} @ D){epsilon}
        compressive = compressive.real
        select_compressive = compressive * selection_mask_m
        return select_compressive

    random_arr = random_flip(mea_shape)
    observ_noise = hparams.noise_level * np.random.randn(noise_shape[0], noise_shape[1])
    
    # Construct mask for inpainting
    if hparams.model_type == 'inpainting':
        if hparams.image_mode == '1D':
            mask = load_mask('Masks', hparams.mask_name_1D)
        elif hparams.image_mode == '2D' or hparams.image_mode == '3D':
            mask = load_mask('Masks', hparams.mask_name_2D)
    elif hparams.model_type == 'denoising' or hparams.model_type == 'compressing':
        mask = None 
    
    # Construct observation 
    if hparams.model_type == 'inpainting':
        #print('mask shape {}'.format(mask.shape))
        y_real = x_real * mask #shape [1, img_wid, img_high, channels]
    elif hparams.model_type == 'denoising' or 'compressing':
        if hparams.type_measurements == 'circulant':
            y_real = circulant_np(x_real.reshape(1,-1)*random_arr, random_vector, selection_mask) + observ_noise 
        elif hparams.type_measurements == 'identity':
            if hparams.image_mode != '3D':
                y_real = np.matmul(x_real.reshape(1,-1), A) + observ_noise #noise level with shape [10,100]
            else:
                y_real = np.reshape(x_real, (1,-1))   ########!!!!!!#####!!!!!!!
        
    # Define decoder network 
    assert hparams.decoder_type == 'deep_image'
    net_fn =  partial(deep_image_prior, 
                    down_layer_count=hparams.down_layer_count, 
                    up_layer_count=hparams.up_layer_count, 
                    num_output_channels=hparams.num_output_channels,
                    image_mode = hparams.image_mode)

    # Fit in
    #print('current number of channels: {}'.format(hparams.k))
    mse, out_img, nparms = fit(net=net_fn,
                           img_shape=x_real.shape,
                           img_name=hparams.img_name, 
                           image_mode=hparams.image_mode,                      
                           type_measurements=hparams.type_measurements,
                           num_measurements=hparams.num_measurements,
                           y_feed=y_real,
                           A_feed=A,
                           mask_info1=hparams.mask_name_1D,
                           mask_feed=mask,
                           ini_channel=hparams.ini_channel,
                           LR=0.005,
                           num_iter=hparams.numit,
                           find_best=True,
                           verbose=True,
                           random_vector=random_vector,
                           selection_mask=selection_mask,
                           save = True,
                           random_array=random_arr)
    #out_img = out_img[0] #4D tensor to 3D tensor if need to plot 
    
    # Compute and print measurement and l2 loss
    measurement_losses = mse[-1]
    l2_losses = get_l2_loss(out_img, x_real, hparams.image_mode)
    print('l2_losses check: {}'.format(l2_losses))
    psnr = 10 * np.log10(1 * 1 / l2_losses) #PSNR
    #print ('Final measurement loss is {}'.format(measurement_losses))
    print ('Final number of params is {}'.format(nparms))
    
    # output 
    if hparams.model_type == 'inpainting':
        if hparams.image_mode == '1D':
            mask_info = hparams.mask_name_1D[8:-4]
        elif hparams.image_mode == '2D' or hparams.image_mode == '3D':
            mask_info = hparams.mask_name_2D[8:-4]
        type_mea_info = 'NA'
        num_mea_info = 'NA'
        noise_level_info = 'NA'
    elif hparams.model_type == 'compressing':
        mask_info = 'NA'
        type_mea_info = hparams.type_measurements
        num_mea_info = hparams.num_measurements
        noise_level_info = 'NA'
    elif hparams.model_type == 'denoising':
        mask_info = 'NA'
        type_mea_info = 'NA'
        num_mea_info = 'NA'
        noise_level_info = hparams.noise_level

    # image saving
    if hparams.save == 1:
        save_out_img(out_img, 'result/', hparams.img_name, hparams.decoder_type, hparams.model_type, num_mea_info, mask_info, noise_level_info, hparams.image_mode)
      
    print ('Final representation PSNR for img_name:{}, model_type:{}, type_mea:{}, num_mea:{}, mask:{}, decoder:{}, up_down_count:{}, initial_channel:{} noise:{} is {}'.format(hparams.img_name, hparams.model_type, type_mea_info, 
            num_mea_info, mask_info, hparams.decoder_type, hparams.down_layer_count, hparams.ini_channel, noise_level_info, psnr))

    if hparams.hyperband_mode == 0:
        pickle_file_path = hparams.pickle_file_path
        if not os.path.exists(pickle_file_path):
            d = {'img_name':[hparams.img_name], 'model_type':[hparams.model_type], 'type_mea':[type_mea_info], 'num_mea':[num_mea_info], 'mask_info':[mask_info], 
                'decoder_type':[hparams.decoder_type], 'up_down_count':[hparams.down_layer_count], 'initial_channel':[hparams.ini_channel], 'noise':[noise_level_info], 'psnr':[psnr]}
            df = pd.DataFrame(data=d)
            df.to_pickle(pickle_file_path)
        else:
            d = {'img_name':hparams.img_name, 'model_type':hparams.model_type, 'type_mea':type_mea_info, 'num_mea':num_mea_info, 'mask_info':mask_info, 
                'decoder_type':hparams.decoder_type, 'up_down_count':hparams.down_layer_count, 'initial_channel':hparams.ini_channel, 'noise':noise_level_info, 'psnr':psnr}
            df = pd.read_pickle(pickle_file_path)
            df = df.append(d, ignore_index=True)
            df.to_pickle(pickle_file_path)
    
    print('END')
    print('\t')



if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images') 
    PARSER.add_argument('--path', type=str, default='Gaussian_signal', help='path stroing the images')
    PARSER.add_argument('--noise_level', type=float, default=0.05, help='std dev of noise') 
    PARSER.add_argument('--img_name', type=str, default='1D_rbf_2.npy', help='image to use') 
    PARSER.add_argument('--model_type', type=str, default='inpainting', help='inverse problem model type') 
    PARSER.add_argument('--type_measurements', type=str, default='circulant', help='measurement type') 
    PARSER.add_argument('--num_measurements', type=int, default=500, help='number of gaussian measurements') 
    PARSER.add_argument('--mask_name_1D', type=str, default='', help='mask to use') 
    PARSER.add_argument('--mask_name_2D', type=str, default='', help='mask to use') 
    PARSER.add_argument('--pickle_file_path', type=str, default='nov30_block_ipt_exp.pkl') 
    PARSER.add_argument('--hyperband_mode', type=int, default=0) 
    PARSER.add_argument('--save', type=int, default=0) 

    # Deep decoder 
    PARSER.add_argument('--decoder_type', type=str, default='deep_image', help='decoder type') 
    PARSER.add_argument('--down_layer_count', type=int, default=5) 
    PARSER.add_argument('--up_layer_count', type=int, default=5) 
    PARSER.add_argument('--num_output_channels', type=int, default=1) 
    PARSER.add_argument('--ini_channel', type=int, default=32) 
    
    # "Training"
    PARSER.add_argument('--rn', type=float, default=0, help='reg_noise_std')
    PARSER.add_argument('--rnd', type=int, default=500, help='reg_noise_decayevery')
    PARSER.add_argument('--numit', type=int, default=10000, help='number of iterations')
   
    HPARAMS = PARSER.parse_args()
    
    main(HPARAMS)