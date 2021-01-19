#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #tunable and depends on the number of available GPUs
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
from compressed_sensing_for_image_signal import * 
import pickle
import itertools


def main_return(hparams, decoder_type, upsample_mode, channel, layer, input_size, filter_size, numit): 
    # Get inputs
    if hparams.image_mode == '1D':
        x_real = np.array(load_1D(hparams.path, hparams.img_name)).astype(np.float32)
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
   
    random_arr = random_flip(4096)
    observ_noise = hparams.noise_level * np.random.randn(noise_shape[0], noise_shape[1])
    
    # Construct mask for inpainting
    if hparams.model_type == 'inpainting':
        if hparams.image_mode == '1D':
            mask = load_mask(hparams.path, hparams.mask_name_1D)
        elif image_mode == '2D' or '3D':
            mask = load_mask(hparams.path, hparams.mask_name_2D)
    elif hparams.model_type == 'denoising' or 'compressing':
        mask = None 
    
    # Construct observation 
    if hparams.model_type == 'inpainting':
        y_real = x_real * mask #shape [1, img_wid, img_high, channels]
    elif hparams.model_type == 'denoising' or 'compressing':
        if hparams.type_measurements == 'circulant':
            y_real = circulant_np(x_real.reshape(1,-1)*random_arr, random_vector, mea_shape, selection_mask) + observ_noise 
        else:
            y_real = np.matmul(x_real.reshape(1,-1), A) + observ_noise #noise level with shape [10,100]
    
    # Define channles*layers 
    channels_times_layers = [channel]*layer  

    # Define activations
    act_function = hparams.activation
    
    # Define decoder network 
    if decoder_type == 'original': 
        net_fn =  partial(decodernw, num_output_channels=x_real.shape[-1], channels_times_layers=channels_times_layers, image_mode=hparams.image_mode, 
                upsample_mode=upsample_mode, factor=hparams.upsample_factor, input_size=input_size, act_fun=act_function)
    elif decoder_type == 'fixed_deconv':  
        net_fn =  partial(fixed_decodernw, num_output_channels=x_real.shape[-1], channels_times_layers=channels_times_layers, image_mode=hparams.image_mode, 
                upsample_mode=upsample_mode, filter_size=filter_size, factor=hparams.upsample_factor, input_size=input_size, act_fun=act_function)
    elif decoder_type == 'deconv':  
        net_fn =  partial(deconv_decoder, num_output_channels=x_real.shape[-1], channels_times_layers=channels_times_layers, image_mode=hparams.image_mode, 
                upsample_mode=upsample_mode, filter_size=filter_size, factor=hparams.upsample_factor, input_size=input_size, act_fun=act_function)

    
    # Fit in
    mse, out_img, nparms = fit(net=net_fn,
                           upsample_factor=hparams.upsample_factor,
                           channels_times_layers=channels_times_layers,
                           img_shape=x_real.shape,
                           image_mode=hparams.image_mode,
                           decoder_type=decoder_type,
                           filter_size=filter_size,
                           upsample_mode=upsample_mode,
                           img_name=hparams.img_name,                           
                           type_measurements=hparams.type_measurements,
                           num_measurements=hparams.num_measurements,
                           num_channels_real = channel,
                           num_layers = layer,
                           act_function = act_function,
                           y_feed=y_real,
                           A_feed=A,
                           mask_info1=hparams.mask_name_1D,
                           mask_info2=hparams.mask_name_2D,
                           mask_feed=mask,
                           LR=0.005,
                           num_iter=numit,
                           find_best=True,
                           verbose=True,
                           input_size=input_size,
                           random_vector=random_vector,
                           selection_mask=selection_mask,
                           save = False,
                           random_array=random_arr)
    #out_img = out_img[0] #4D tensor to 3D tensor if need to plot 
    
    # Compute PSNR
    l2_losses = get_l2_loss(out_img, x_real)
    psnr = 10 * np.log10(1 * 1 / l2_losses) #PSNR
    
    # Print result 
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
        factor_record = round(1 * pow(4096/input_size, 1/layer), 4)
    
    print ('Final representation PSNR for img_name:{}, model_type:{}, type_mea:{}, num_mea:{}, mask:{}, decoder:{}, upsample_mode:{}, filter_size:{}, #channels:{} #layers:{} input_size:{} up_factor:{} is {}'.format(
            hparams.img_name, hparams.model_type, type_mea_info, num_mea_info, mask_info, decoder_type, upsample_mode, filter_size, channel, layer, input_size, factor_record, psnr))
    
    # Return result 
    return psnr


def get_random_hyperparameter_configuration(success_halving=True):
    channel_list = [80,100,120,150,170,200,230,250,280]
    layer_list = [20,42,45,47,50,52]
    input_size_list = [14,17,20,23,26,29,32,35,50,70]
    filter_size_list = [4] 
    #model_type_list = [('original','bilinear'),('fixed_deconv','bilinear'),('fixed_deconv','none'),('deconv','bilinear'),('deconv','none')] #5
    configurations = list(itertools.product(channel_list, layer_list, input_size_list, filter_size_list)) #4*7*7*5 = 980
    if success_halving:
        return configurations#[(32,3,64,3), ...]
    else:
        return channel_list, layer_list, input_size_list, filter_size_list
    
    
def run_then_return_val_loss(hparams, num_iters, hyperparameters):
    psnr = main_return(hparams, 'fixed_deconv', 'bilinear', hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3], num_iters)
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
        # if i == s-1 and int(n_i/eta) >= 2:
        #     T = T[0]
        #with open(hparams.pickle_file_path, 'wb') as f:
        #    pickle.dump(T, f)
        print('#config is {}, #iters is {}, loss criteria is {}'.format(int(n_i), int(r_i), val_PSNR[descending_sort_idx[int(n_i/eta)-1]]))
    
    return T #[(32,3,64,3]
    

def greedy(hparams, num_iters=8000):
    hyperparameters = get_random_hyperparameter_configuration(success_halving=False)
    decoder_type, upsample_mode, chn_lst, lay_list, ipt_list, fil_list = 'fixed_deconv', 'bilinear', hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3]
    baseline = main_return(hparams, decoder_type, upsample_mode, chn_lst[0], lay_list[0], ipt_list[0], fil_list[0], num_iters)
    chn_per, lay_per, ipt_per, fil_per = [], [], [], []

    for chn in chn_lst[1:]:
        psnr = main_return(hparams, decoder_type, upsample_mode, chn, lay_list[0], ipt_list[0], fil_list[0], num_iters)
        diff = psnr - baseline 
        chn_per.append(diff)

    for lay in lay_list[1:]:
        psnr = main_return(hparams, decoder_type, upsample_mode, chn_lst[0], lay, ipt_list[0], fil_list[0], num_iters)
        diff = psnr - baseline 
        lay_per.append(diff)

    for ipt in ipt_list[1:]:
        psnr = main_return(hparams, decoder_type, upsample_mode, chn_lst[0], lay_list[0], ipt, fil_list[0], num_iters)
        diff = psnr - baseline 
        ipt_per.append(diff)

    for fil in fil_list[1:]:
        psnr = main_return(hparams, decoder_type, upsample_mode, chn_lst[0], lay_list[0], ipt_list[0], fil, num_iters)
        diff = psnr - baseline 
        fil_per.append(diff)

    print(hparams.img_name)
    print('chn: ', chn_per)
    print('lay: ', lay_per)
    print('ipt: ', ipt_per)
    print('fil: ', fil_per)
    print('End')
    print('\n')
    
        
if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images') ###################################
    PARSER.add_argument('--path', type=str, default='Gaussian_signal', help='path stroing the images')
    PARSER.add_argument('--noise_level', type=float, default=0.00, help='std dev of noise') ###################################
    PARSER.add_argument('--img_name', type=str, default='1D_rbf_2.npy', help='image to use') ###################################
    PARSER.add_argument('--model_type', type=str, default='inpainting', help='inverse problem model type') ##################################
    PARSER.add_argument('--type_measurements', type=str, default='circulant', help='measurement type') ###################################
    PARSER.add_argument('--num_measurements', type=int, default=500, help='number of gaussian measurements') ###################################
    PARSER.add_argument('--mask_name_1D', type=str, default='', help='mask to use') ###################################
    PARSER.add_argument('--mask_name_2D', type=str, default='', help='mask to use') ###################################
    PARSER.add_argument('--pickle_file_path', type=str, default='hyperband_rsl.pkl') #######################################
    PARSER.add_argument('--hyperband_mode', type=int, default=1) #######################################
    PARSER.add_argument('--selection_type', type=str, default='successive_halving') #######################################
    PARSER.add_argument('--output_file_path', type=str, default='dec14_circulant.txt') #######################################

    # Deep decoder 
    PARSER.add_argument('--k', type=int, default=256, help='number of channel dimension') ###################################
    PARSER.add_argument('--num_channel', type=int, default=6, help='number of upsample channles') ###################################
    PARSER.add_argument('--decoder_type', type=str, default='fixed_deconv', help='decoder type') ###################################
    PARSER.add_argument('--upsample_mode', type=str, default='bilinear', help='upsample type') ###################################
    PARSER.add_argument('--filter_size', type=int, default=4, help='upsample type') ###################################
    PARSER.add_argument('--upsample_factor', type=float, default=None, help='upsample factor') ###################################
    PARSER.add_argument('--input_size', type=int, default=128, help='input_size') ###################################
    PARSER.add_argument('--activation', type=str, default='relu', help='activation type') ###################################
    
    # "Training"
    PARSER.add_argument('--rn', type=float, default=0, help='reg_noise_std')
    PARSER.add_argument('--rnd', type=int, default=500, help='reg_noise_decayevery')
    PARSER.add_argument('--numit', type=int, default=10000, help='number of iterations')
    
    HPARAMS = PARSER.parse_args()

    file = sys.stdout   
    sys.stdout = open(HPARAMS.output_file_path, 'w')         
    
    if HPARAMS.selection_type == 'successive_halving':
        out_conig = successive_halving(HPARAMS)
        #out_conig = hyperband(HPARAMS)
        if len(out_conig) > 1:
            out_conig = out_conig[0]
        if HPARAMS.model_type == 'denoising':
            task_info = HPARAMS.img_name + '/' + HPARAMS.type_measurements
        else:
            task_info = HPARAMS.img_name + '/' + HPARAMS.mask_name_1D
        print('Best Config of {}: #channel is {}, #layer is {}, input_size is {}, filter_size is {}'.format(task_info, out_conig[0][0], 
            out_conig[0][1], out_conig[0][2], out_conig[0][3]))

    else:
        greedy(HPARAMS)

