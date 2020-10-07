#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
import tensorflow as tf
#print('Tensorflow version', tf.__version__)
from functools import partial
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
#from skimage.color import rgb2gray, rgba2rgb
#from matplotlib import pyplot as plt
from argparse import ArgumentParser
import math
import pandas as pd 
from compressed_sensing_for_image_signal import *
import pickle
import itertools


def main_return(hparams, channel, layer, input_size, filter_size, numit):
    # Get inputs
    if hparams.image_mode == '1D':
        x_real = np.array(load_1D(hparams.path, hparams.img_name)).astype(np.float32)
    elif hparams.image_mode == '2D':
        x_real = np.array(load_2D(hparams.path, hparams.img_name)).astype(np.float32)
    else:
        x_real = np.array(load_img(hparams.path, hparams.img_name)).astype(np.float32)
    
    # Construct measurements/noises for compressed sensing 
    mea_shape = x_real.shape[1] * x_real.shape[2] * x_real.shape[3] #i.e. img_wid*img_wid*3
    if hparams.type_measurements == 'random':
        A = np.random.randn(mea_shape, hparams.num_measurements).astype(np.float32)
        noise_shape = [1, hparams.num_measurements]
    elif hparams.type_measurements == 'identity':
        A = np.identity(mea_shape).astype(np.float32)
        noise_shape = [1, mea_shape]
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
        y_real = np.matmul(x_real.reshape(1,-1), A) + observ_noise #noise level with shape [10,100]
    #y_name = hparams.image_mode + '_y_' + hparams.decoder_type + '_' + str(hparams.filter_size) + '_' + hparams.upsample_mode + '.npy'
    #np.save(os.path.join('result/', y_name), y_real[0,:,:,0])
    
    # Define num_channles 
    num_channels = [channel]*layer  
    
    # Define decoder network 
    if hparams.decoder_type == 'original': 
        net_fn =  partial(decodernw, num_output_channels=x_real.shape[-1], num_channels_up=num_channels, 
                image_mode=hparams.image_mode, upsample_mode=hparams.upsample_mode, factor=hparams.upsample_factor, input_size=input_size)
    elif hparams.decoder_type == 'fixed_deconv':  
        net_fn =  partial(fixed_decodernw, num_output_channels=x_real.shape[-1], num_channels_up=num_channels, 
                image_mode=hparams.image_mode, upsample_mode=hparams.upsample_mode, filter_size=filter_size, factor=hparams.upsample_factor, input_size=input_size)
    elif hparams.decoder_type == 'deconv':  
        net_fn =  partial(deconv_decoder, num_output_channels=x_real.shape[-1], num_channels_up=num_channels, 
                image_mode=hparams.image_mode, upsample_mode=hparams.upsample_mode, filter_size=filter_size, factor=hparams.upsample_factor, input_size=input_size)

    
    # Fit in
    mse, out_img, nparms = fit(net=net_fn,
                           upsample_factor=hparams.upsample_factor,
                           num_channels=num_channels,
                           img_shape=x_real.shape,
                           image_mode=hparams.image_mode,
                           decoder_type=hparams.decoder_type,
                           filter_size=filter_size,
                           upsample_mode=hparams.upsample_mode,
                           img_name=hparams.img_name,                           
                           type_measurements=hparams.type_measurements,
                           num_measurements=hparams.num_measurements,
                           num_channels_real = channel,
                           num_channel = layer,
                           y_feed=y_real,
                           A_feed=A,
                           mask_info1=hparams.mask_name_1D,
                           mask_info2=hparams.mask_name_2D,
                           mask_feed=mask,
                           LR=0.005,
                           num_iter=numit,
                           find_best=True,
                           verbose=True,
                           input_size=input_size)
    #out_img = out_img[0] #4D tensor to 3D tensor if need to plot 
    
    # Compute PSNR
    l2_losses = get_l2_loss(out_img, x_real)
    psnr = 10 * np.log10(1 * 1 / l2_losses) #PSNR
    
    # Print result 
    if hparams.image_mode == '1D':
        mask_info = hparams.mask_name_1D[8:-4]
    elif image_mode == '2D' or '3D':
        mask_info = hparams.mask_name_2D[8:-4]
    if hparams.upsample_factor:
        factor_record = hparams.upsample_factor
    else:
        factor_record = round(1 * pow(4096/input_size, 1/(layer-1)), 2)
    print ('img_name:{}, mask_info:{}, decoder_type:{}, filter_size:{}, upsample_mode:{}, #channels:{} #layers:{} up_factor:{} input_size:{} WITH FINAL PSNR AS {}'.format(hparams.img_name, mask_info, 
        hparams.decoder_type, filter_size, hparams.upsample_mode, channel, layer, factor_record, input_size, psnr))
    
    # Return result 
    return psnr


def get_random_hyperparameter_configuration(success_halving=False):
    channel_list = [32,64,128,256,368,512] #6
    layer_list = [3,4,5,6,7,8,9,10,11,12,13,14] #12
    input_size_list = [64,128,256,384,512,640,768,896,1024] #9
    filter_size_list = [3,4,5,8,16] #5
    #configurations = list(itertools.product(channel_list, layer_list, input_size_list)) #6*12*9 = 648 
    configurations = list(itertools.product(channel_list, layer_list, input_size_list, filter_size_list)) #6*12*9*5 = 3240 
    if not success_halving:
        num_len = len(channel_list)*len(layer_list)*len(input_size_list)
        idx = np.random.randint(1,num_len)
        output_configuration = output_configurations[idx]
        return output_configuration #(32,3,64)
    else:
        return configurations#[(32,3,64,3), ...]
    
    
def run_then_return_val_loss(hparams, num_iters, hyperparameters):
    psnr = main_return(hparams, hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3], num_iters)
    return psnr 

def successive_halving(hparams, eta=3, max_iter=8192):   
    T = get_random_hyperparameter_configuration(success_halving=True)
    n = len(T) # initial number of configurations
    logeta = lambda x: math.log(x)/math.log(eta)
    s = int(logeta(n)) # number of unique executions of Successive Halving (minus 1)
    r = max_iter*2**(-(s-1)) # initial number of iterations to run configurations for       # stablise eta to be 2 for r
    
    for i in range(s):
        n_i = n*eta**(-i)
        r_i = r*2**(i) # stablise eta to be 2 for r 
        val_PSNR = [ run_then_return_val_loss(hparams, num_iters=int(r_i), hyperparameters=t) for t in T ]
        ascending_sort_idx = np.argsort(val_PSNR)
        descending_sort_idx = ascending_sort_idx[::-1]
        T = [ T[i] for i in descending_sort_idx[0:int( n_i/eta )] ]
        #with open(hparams.pickle_file_path, 'wb') as f:
        #    pickle.dump(T, f)
        print('#config is {}, #iters is {}, loss criteria is {}'.format(int(n_i), int(r_i), val_PSNR[descending_sort_idx[int(n_i/eta)-1]]))
    
    return T #[(32,3,64,3)]
    

def hyperband(hparams, eta=2, max_iter=8192):
    logeta = lambda x: math.log(x)/math.log(eta)
    max_config = 648
    s_max = int(logeta(max_config))  # number of unique executions of Successive Halving (minus one)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    best_score = -10000
    for s in reversed(range(s_max)):
        n = max_config*eta**(-(s_max-s-1)) * s_max/s+1 # initial number of configurations
        r = max_iter*eta**(-(s)) # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        if s == s_max-1:
            T = get_random_hyperparameter_configuration(success_halving=True)
        else:
            T = [ get_random_hyperparameter_configuration() for i in range(n) ] 
        
        i = 0
        while len(T) != 1:
            n_i = n*eta**(-i)
            r_i = r*eta**(i)
            val_losses = [ run_then_return_val_loss(hparams, num_iters=int(r_i), hyperparameters=t) for t in T ]
            losses_sort_idx = np.argsort(val_losses)
            T = [ T[i] for i in losses_sort_idx[0:int( n_i/eta )] ]
            #with open(hparams.pickle_file_path, 'wb') as f:
            #    pickle.dump(T, f)
            print('idx of success halving is {}, #total config is {}, #config is {}, #iters is {}, loss criteria is {}'.format(s, n, 
                    n_i, r_i, val_losses[losses_sort_idx[int(n_i/eta)-1]]))
            i += 1
        #### End Finite Horizon Successive Halving with (n,r)
        if val_losses[losses_sort_idx[int(n_i/eta)-1]] > best_score:
            best_score = val_losses[losses_sort_idx[int(n_i/eta)-1]]
            best_config = T
             
    return best_config
    
        
if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images')  
    PARSER.add_argument('--path', type=str, default='', help='path stroing the images')
    PARSER.add_argument('--noise_level', type=float, default=0.05, help='std dev of noise') 
    PARSER.add_argument('--img_name', type=str, default='1D_rbf_2.npy', help='image to use') ###
    PARSER.add_argument('--model_type', type=str, default='inpainting', help='inverse problem model type') 
    PARSER.add_argument('--mask_name_1D', type=str, default='', help='mask to use') ###
    PARSER.add_argument('--mask_name_2D', type=str, default='', help='mask to use') 
    PARSER.add_argument('--pickle_file_path', type=str, default='config_list_sep_28.pkl') 
    
    # Measurement type specific hparams
    PARSER.add_argument('--type_measurements', type=str, default='identity', help='measurement type') 
    PARSER.add_argument('--num_measurements', type=int, default=500, help='number of gaussian measurements') 
    
    # "Training"
    PARSER.add_argument('--rn', type=float, default=0, help='reg_noise_std')
    PARSER.add_argument('--rnd', type=int, default=500, help='reg_noise_decayevery')
    #PARSER.add_argument('--numit', type=int, default=10000, help='number of iterations')

    # Deep decoder 
    PARSER.add_argument('--decoder_type', type=str, default='fixed_deconv', help='decoder type') ##############
    PARSER.add_argument('--upsample_mode', type=str, default='bilinear', help='upsample type') ##############
    PARSER.add_argument('--filter_size', type=int, default=4, help='upsample type') #################
    PARSER.add_argument('--upsample_factor', type=float, default=None, help='upsample factor')
    
    HPARAMS = PARSER.parse_args()       
    
    out_conig = successive_halving(HPARAMS)
    #out_conig = hyperband(HPARAMS)
    assert len(out_conig) == 1
    print('Best Config: kernel_size is {}, channel is {}, layer is {}, input_size is {}, up-factor is {}'.format(out_conig[0][3], out_conig[0][0], out_conig[0][1], out_conig[0][2], round(1 * pow(4096/out_conig[0][2], 1/(out_conig[0][0]-1)), 2)))