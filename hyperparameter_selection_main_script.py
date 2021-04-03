#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np
#print('Tensorflow version', tf.__version__)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from functools import partial
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
from argparse import ArgumentParser
import math
import pandas as pd 
from compressed_sensing_for_image_signal import * 
import pickle
import itertools
import sys



def main_return(hparams, channel, layer, input_size, filter_size, lr, numit, decoder_type='fixed_deconv', upsample_mode='bilinear'): 
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
        
    def circulant_np(signal_vector, random_vector, selection_mask):
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
        y_real = x_real * mask #shape [1, img_wid, img_high, channels]
    elif hparams.model_type == 'denoising' or hparams.model_type == 'compressing':
        if hparams.type_measurements == 'circulant':
            y_real = circulant_np(x_real.reshape(1,-1)*random_arr, random_vector, selection_mask) #+ observ_noise 
        elif hparams.type_measurements == 'identity':
            if hparams.image_mode != '3D':
                y_real = np.matmul(x_real.reshape(1,-1), A) + observ_noise #noise level with shape [10,100]
            else:
                y_real = np.reshape(x_real, (1,-1))   ########!!!!!!#####!!!!!!!
    
    # Define channles*layers 
    channels_times_layers = [channel]*(layer-1)  

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
                           LR=lr,
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
        noise_level_info = '005'
        
    if hparams.upsample_factor:
        factor_record1 = hparams.upsample_factor
        factor_record2 = hparams.upsample_factor
    else:
        factor_record1 = round(1 * pow(x_real.shape[1]/input_size, 1/(layer-1)), 4) ##3D
        factor_record2 = round(1 * pow(x_real.shape[2]/input_size, 1/(layer-1)), 4) ##3D
    
    print ('Final representation PSNR for img_name:{}, model_type:{}, type_mea:{}, num_mea:{}, mask:{}, noise:{}, decoder:{}, upsample_mode:{}, filter_size:{}, #channels:{} #layers:{} input_size:{} up_factor1:{} up_factor2:{} step_size:{} is {}'.format(
            hparams.img_name, hparams.model_type, type_mea_info, num_mea_info, mask_info, noise_level_info, decoder_type, upsample_mode, filter_size, channel, layer, input_size, factor_record1, factor_record2, lr, psnr))
    
    # Return result 
    return psnr


def get_random_hyperparameter_configuration(hparams, ratio=0.1, success_halving=True):
    chn_ls = hparams.chn_ls
    lay_ls = hparams.lay_ls
    ipt_ls = hparams.ipt_ls
    fit_ls = hparams.fit_ls
    lr_ls = hparams.lr_ls
    #model_type_list = [('original','bilinear'),('fixed_deconv','bilinear'),('fixed_deconv','none'),('deconv','bilinear'),('deconv','none')] #5
    configurations = list(itertools.product(chn_ls, lay_ls, ipt_ls, fit_ls, lr_ls)) #4*7*7*5 = 980
    random.shuffle(configurations)
    configurations_select = configurations[:int(len(configurations)*ratio)]
    if success_halving:
        return configurations_select#[(32,3,64,3), ...]
    else:
        return chn_ls, lay_ls, ipt_ls, fit_ls, lr_ls
    
    
def run_then_return_val_loss(hparams, num_iters, hyperparameters):
    psnr = main_return(hparams, hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3], hyperparameters[4], num_iters)
    return psnr 


# np.exp( np.log(self.r/float(self.ressource_unit))/math.floor(np.log(len(T))/np.log(2.)) )


def successive_halving(hparams, eta=3, max_iter=2048):   
    T = get_random_hyperparameter_configuration(hparams, ratio=1, success_halving=True)
    n = len(T) # initial number of configurations
    logeta = lambda x: math.log(x)/math.log(eta)
    s = int(logeta(n)) # number of unique executions of Successive Halving (minus 1)
    r = max_iter*eta**(-(s-1)) # initial number of iterations to run configurations for       # stablise eta to be 2 for r
    
    for i in range(s):
        n_i = n*eta**(-i)
        r_i = r*eta**(i) # stablise eta to be 2 for r 
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
    



def greedy(hparams, num_iters=6000):
    chn_ls = [32,64,128,192,256,320,384,448,512]
    lay_ls = [2,4,6,8,10,14,18,22,26,32]
    ipt_ls = [2,4,8,16,24,32,48,128,256,512]
    fit_ls = [4,8,16,32,48,64]
    lrn_ls = [0.0005,0.0010,0.0015,0.0020,0.0025,0.0030,0.0035,0.0040,0.0045,0.0050,0.0055,0.0060,0.0065,0.0070,0.0075,0.0080,0.0085,0.0090,0.0095,
                0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050]

    #Step 1: select initial potential values
    configuration_state = {}

    if hparams.chn != chn_ls[0] and hparams.chn != chn_ls[-1]:
        anchor = min(chn_ls, key=lambda x:abs(x-hparams.chn))
        if hparams.chn < anchor:
            chn_ab = [ chn_ls[chn_ls.index(anchor)-1], hparams.chn, anchor ]
        if hparams.chn == anchor:
            chn_ab = [ chn_ls[chn_ls.index(hparams.chn)-1], hparams.chn, chn_ls[chn_ls.index(hparams.chn)+1] ]
        if hparams.chn > anchor:
            chn_ab = [ anchor, hparams.chn, chn_ls[chn_ls.index(anchor)+1] ]
        configuration_state['chn'] = 1
    else:
        chn_ab = [ hparams.chn, hparams.chn, hparams.chn ]
        configuration_state['chn'] = 0
    
    if hparams.lay != lay_ls[0] and hparams.lay != lay_ls[-1]:
        anchor = min(lay_ls, key=lambda x:abs(x-hparams.lay))
        if hparams.lay < anchor:
            lay_ab = [ lay_ls[lay_ls.index(anchor)-1], hparams.lay, anchor ]
        if hparams.lay == anchor:
            lay_ab = [ lay_ls[lay_ls.index(hparams.lay)-1], hparams.lay, lay_ls[lay_ls.index(hparams.lay)+1] ]
        if hparams.lay > anchor:
            lay_ab = [ anchor, hparams.lay, lay_ls[lay_ls.index(anchor)+1] ]
        configuration_state['lay'] = 1
    else:
        lay_ab = [ hparams.lay, hparams.lay, hparams.lay ]
        configuration_state['lay'] = 0
    
    if hparams.ipt != ipt_ls[0] and hparams.ipt != ipt_ls[-1]:
        anchor = min(ipt_ls, key=lambda x:abs(x-hparams.ipt))
        if hparams.ipt < anchor:
            ipt_ab = [ ipt_ls[ipt_ls.index(anchor)-1], hparams.ipt, anchor ]
        if hparams.ipt == anchor:
            ipt_ab = [ ipt_ls[ipt_ls.index(hparams.ipt)-1], hparams.ipt, ipt_ls[ipt_ls.index(hparams.ipt)+1] ]
        if hparams.ipt > anchor:
            ipt_ab = [ anchor, hparams.ipt, ipt_ls[ipt_ls.index(anchor)+1] ]
        configuration_state['ipt'] = 1
    else:
        ipt_ab = [ hparams.ipt, hparams.ipt, hparams.ipt ]
        configuration_state['ipt'] = 0
    
    if hparams.fit != fit_ls[0] and hparams.fit != fit_ls[-1]:
        anchor = min(fit_ls, key=lambda x:abs(x-hparams.fit))
        if hparams.fit < anchor:
            fit_ab = [ fit_ls[fit_ls.index(anchor)-1], hparams.fit, anchor ]
        if hparams.fit == anchor:
            fit_ab = [ fit_ls[fit_ls.index(hparams.fit)-1], hparams.fit, fit_ls[fit_ls.index(hparams.fit)+1] ]
        if hparams.fit > anchor:
            fit_ab = [ anchor, hparams.fit, fit_ls[fit_ls.index(anchor)+1] ]
        configuration_state['fit'] = 1
    else:
        fit_ab = [ hparams.fit, hparams.fit, hparams.fit ]
        configuration_state['fit'] = 0
    
    anchor = min(lrn_ls, key=lambda x:abs(x-hparams.lrn))
    lrn_ab = [ anchor, anchor, anchor ]
    configuration_state['lrn'] = 0

    configuration_dic = {'chn':chn_ab, 'lay':lay_ab, 'ipt':ipt_ab, 'fit':fit_ab, 'lrn':lrn_ab}

    #Step 2: fine tuning
    def g_return(j, hp_name, config):
        if j == 0:
            value = int((config[hp_name][0] + config[hp_name][1])/2)
        elif j == 2:
            value = int((config[hp_name][1] + config[hp_name][2])/2)
        else:
            value = config[hp_name][1]
        
        if hp_name == 'chn':
            r = main_return(hparams, channel=value, layer=config['lay'][1], input_size=config['ipt'][1], filter_size=config['fit'][1], lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'lay':
            r = main_return(hparams, channel=config['chn'][1], layer=value, input_size=config['ipt'][1], filter_size=config['fit'][1], lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'ipt':
            r = main_return(hparams, channel=config['chn'][1], layer=config['lay'][1], input_size=value, filter_size=config['fit'][1], lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'fit':
            r = main_return(hparams, channel=config['chn'][1], layer=config['lay'][1], input_size=config['ipt'][1], filter_size=value, lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'lrn':
            r = main_return(hparams, channel=config['chn'][1], layer=config['lay'][1], input_size=config['ipt'][1], filter_size=config['fit'][1], lr=value, numit=num_iters)
        
        return r


    # configuration_state = {'chn':0, 'lay':1, 'ipt':1, 'fit':0, 'lrn':0}
    round_count = 0
    while sum(list(configuration_state.values())) != 0:
        print(configuration_state)
        print('prior info of round {}: #chn {}, #lay {}, ipt {}, fit {}, lrn {}'.format(
                round_count, configuration_dic['chn'][1], configuration_dic['lay'][1], configuration_dic['ipt'][1], configuration_dic['fit'][1], configuration_dic['lrn'][1]))
        for hp_name in list(configuration_dic.keys()): 
            if configuration_state[hp_name] == 0:
                continue
            elif configuration_dic[hp_name][0] == configuration_dic[hp_name][1] or configuration_dic[hp_name][1] == configuration_dic[hp_name][2]:
                configuration_state[hp_name] = 0
                continue
            else:
                configuration_dic_copy = configuration_dic[hp_name].copy()
                psnr_dic = {}
                for j in range(len(configuration_dic[hp_name])):
                    psnr_dic[str(j)] = g_return(j, hp_name, configuration_dic)
                max_key = int(max(psnr_dic, key=psnr_dic.get))
                if max_key == 0:
                    configuration_dic[hp_name] = [ configuration_dic_copy[0], int((configuration_dic_copy[0]+configuration_dic_copy[1])/2), configuration_dic_copy[1] ]
                elif max_key == 1:
                    configuration_dic[hp_name] = [ configuration_dic_copy[1] , configuration_dic_copy[1], configuration_dic_copy[1] ]
                elif max_key == 2:
                    configuration_dic[hp_name] = [ configuration_dic_copy[1], int((configuration_dic_copy[1]+configuration_dic_copy[2])/2), configuration_dic_copy[2] ]
            # print('for loop result: #channel is {}, #layer is {}, input_size is {}, filter_size is {}, step_size is {}'.format(configuration_dic['chn'][1], configuration_dic['lay'][1], configuration_dic['ipt'][1], configuration_dic['fit'][1], configuration_dic['lrn'][1]))
        print('result of round {}: #chn {}, #lay {}, ipt {}, fit {}, lrn {}'.format(
                round_count, configuration_dic['chn'][1], configuration_dic['lay'][1], configuration_dic['ipt'][1], configuration_dic['fit'][1], configuration_dic['lrn'][1]))
        #print(configuration_state)
        print('\n')
        round_count += 1
    print('fine tune completed')
    return round_count, configuration_dic

        
if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images') 
    PARSER.add_argument('--path', type=str, default='Gaussian_signal', help='path stroing the images')
    PARSER.add_argument('--noise_level', type=float, default=0.00, help='std dev of noise') 
    PARSER.add_argument('--img_name', type=str, default='1D_rbf_2.npy', help='image to use') 
    PARSER.add_argument('--model_type', type=str, default='inpainting', help='inverse problem model type')
    PARSER.add_argument('--type_measurements', type=str, default='circulant', help='measurement type') 
    PARSER.add_argument('--num_measurements', type=int, default=500, help='number of gaussian measurements') 
    PARSER.add_argument('--mask_name_1D', type=str, default='', help='mask to use') 
    PARSER.add_argument('--mask_name_2D', type=str, default='', help='mask to use') 
    PARSER.add_argument('--pickle_file_path', type=str, default='hyperband_rsl.pkl') 
    PARSER.add_argument('--hyperband_mode', type=int, default=1) 
    PARSER.add_argument('--selection_type', type=str, default='successive_halving') 
    PARSER.add_argument('--output_file_path', type=str, default='dec14_circulant.txt') 

    # Deep decoder 
    PARSER.add_argument('--k', type=int, default=256, help='number of channel dimension') 
    PARSER.add_argument('--num_channel', type=int, default=6, help='number of upsample channles') 
    PARSER.add_argument('--decoder_type', type=str, default='fixed_deconv', help='decoder type') 
    PARSER.add_argument('--upsample_mode', type=str, default='bilinear', help='upsample type') 
    PARSER.add_argument('--filter_size', type=int, default=4, help='upsample type') 
    PARSER.add_argument('--upsample_factor', type=float, default=None, help='upsample factor') 
    PARSER.add_argument('--input_size', type=int, default=128, help='input_size') 
    PARSER.add_argument('--activation', type=str, default='relu', help='activation type') 
    
    # "Training"
    PARSER.add_argument('--rn', type=float, default=0, help='reg_noise_std')
    PARSER.add_argument('--rnd', type=int, default=500, help='reg_noise_decayevery')
    PARSER.add_argument('--numit', type=int, default=10000, help='number of iterations')

    # Selection
    PARSER.add_argument('--chn_ls', type=int, nargs='+', default=[32,64,128,192,256,320,384,448,512], help='a list of channels')
    PARSER.add_argument('--lay_ls', type=int, nargs='+', default=[2,4,6,8,10,14,18,22,26,32], help='a list of layers')
    PARSER.add_argument('--ipt_ls', type=int, nargs='+', default=[2,4,8,16,24,32,48,128,256,512], help='a list of input_size')
    PARSER.add_argument('--fit_ls', type=int, nargs='+', default=[4,8,16,32,48,64], help='a list of filters')
    PARSER.add_argument('--lr_ls', type=float, nargs='+', default=[0.0005,0.001,0.003,0.005,0.007,0.01,0.002,0.025,0.05], help='a list of learning rate')

    PARSER.add_argument('--chn', type=int, default=32, help='channels')
    PARSER.add_argument('--lay', type=int, default=32, help='layers')
    PARSER.add_argument('--ipt', type=int, default=32, help='input_size')
    PARSER.add_argument('--fit', type=int, default=32, help='filters')
    PARSER.add_argument('--lrn', type=float, default=0.005, help='learning rate')
    
    HPARAMS = PARSER.parse_args()

    file = sys.stdout   
    sys.stdout = open(HPARAMS.output_file_path, 'w')         
    
    if HPARAMS.model_type == 'denoising' or HPARAMS.model_type == 'compressing':
        task_info = HPARAMS.img_name + '/' + HPARAMS.type_measurements
    else:
        task_info = HPARAMS.img_name + '/' + HPARAMS.mask_name_1D
    
    if HPARAMS.selection_type == 'successive_halving':
        out_conig = successive_halving(HPARAMS)
        #out_conig = hyperband(HPARAMS)
        if len(out_conig) > 1:
            out_conig = out_conig[0]
        print('Best Config of {}: #channel is {}, #layer is {}, input_size is {}, filter_size is {}, step_size is {}'.format(task_info, out_conig[0][0], 
            out_conig[0][1], out_conig[0][2], out_conig[0][3], out_conig[0][4]))
    else:
        round_count, final_dict = greedy(HPARAMS)
        print('After {} rounds, fine-tuned Config of {}: #channel is {}, #layer is {}, input_size is {}, filter_size is {}, step_size is {}'.format(round_count, task_info, final_dict['chn'][1], 
            final_dict['lay'][1], final_dict['ipt'][1], final_dict['fit'][1], final_dict['lrn'][1]))
