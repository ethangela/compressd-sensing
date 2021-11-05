from __future__ import print_function
# import matplotlib.pyplot as plt
#%matplotlib notebook

import os
from argparse import ArgumentParser
import sigpy.mri as mr

import sigpy as sp
import sigpy.mri as mr
from os import listdir
from os.path import isfile, join

import warnings
warnings.filterwarnings('ignore')

from include import *

from PIL import Image
import PIL
import h5py
#from skimage.metrics import structural_similarity as ssim
from common.evaluate import *
from pytorch_msssim import ms_ssim
import pickle
from common.subsample import MaskFunc
import itertools
import math
import urllib.request as urll

from DIP_UNET_models.skip import *

import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
#from models import *
#from utils.denoising_utils import *

# from facebook MRI
#import transforms
from include import transforms as transform

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # gpu = 2
    # torch.cuda.set_device(gpu)
    print("num GPUs",torch.cuda.device_count())
    print(torch.cuda.is_available())
else:
    dtype = torch.FloatTensor




def fit_untrained(parnet, num_channels, num_layers, mask2d_, mask, in_size, slice_ksp, slice_ksp_torchtensor, LR=0.008, num_iter=20000):
    ### fixing the scaling (note that it can be done using the under-sampled kspace as well, but we do it using the full kspace)
    scale_out = 1
    scaling_factor,ni = get_scale_factor(parnet,
                                       num_channels,
                                       in_size,
                                       slice_ksp,
                                       scale_out=scale_out)
    slice_ksp_torchtensor = slice_ksp_torchtensor * scaling_factor
    slice_ksp = slice_ksp * scaling_factor
    ### mask the ksapce
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor, mask = mask) #(15, 640, 368)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    
    measurement = ksp2measurement(slice_ksp).type(dtype)
    lsimg = lsreconstruction(measurement)
    
    ### fit the network to the under-sampled measurement
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))
    scale_out,sover,pover,norm_ratio,par_mse_n, par_mse_t, parni, parnet = fit( in_size = in_size,
                                                                num_channels=[num_channels]*(num_layers-1),
                                                                num_iter=num_iter,
                                                                LR=LR,
                                                                mask = mask2d_,
                                                                apply_f = forwardm,
                                                                img_noisy_var=unders_measurement,
                                                                net=parnet,
                                                                upsample_mode="free",
                                                                img_clean_var=Variable(lsest).type(dtype),
                                                                #lsimg = lsimg,
                                                                find_best=True,
                                                                loss_type="MSE",
                                                                scale_out=scale_out,
                                                                net_input = ni,
                                                                OPTIMIZER = "adam"
                                                                          )
    return parnet, parni, slice_ksp


def data_consistency(parnet, parni, mask1d, slice_ksp):
    img = parnet(parni.type(dtype))
    s = img.shape
    ns = int(s[1]/2) # number of slices
    fimg = Variable( torch.zeros( (s[0],ns,s[2],s[3],2 ) ) ).type(dtype)
    for i in range(ns):
        fimg[0,i,:,:,0] = img[0,2*i,:,:]
        fimg[0,i,:,:,1] = img[0,2*i+1,:,:]
    Fimg = transform.fft2(fimg) # dim: (1,num_slices,x,y,2)
    # ksp has dim: (num_slices,x,y)
    meas = ksp2measurement(slice_ksp) # dim: (1,num_slices,x,y,2)
    mask = torch.from_numpy(np.array(mask1d, dtype=np.uint8))
    ksp_dc = Fimg.clone()
    ksp_dc = ksp_dc.detach().cpu()
    ksp_dc[:,:,:,mask==1,:] = meas[:,:,:,mask==1,:] # after data consistency block

    img_dc = transform.ifft2(ksp_dc)[0]
    out = []
    for img in img_dc.detach().cpu():
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]

    par_out_chs = np.array(out)
    #par_out_chs = parnet( parni.type(dtype),scale_out=scale_out ).data.cpu().numpy()[0]
    par_out_imgs = channels2imgs(par_out_chs)

    prec = crop_center2(root_sum_of_squares2(par_out_imgs),320,320)
    return prec


def scores(im1,im2):
    im1 = (im1-im1.mean()) / im1.std()
    im1 *= im2.std()
    im1 += im2.mean()
    
    vif_ = vifp_mscale(im1,im2,sigma_nsq=im1.mean())
    
    ssim_ = ssim(np.array([im1]), np.array([im2]))
    psnr_ = psnr(np.array([im1]),np.array([im2]))

    dt = torch.FloatTensor
    im11 = torch.from_numpy(np.array([[im1]])).type(dt)
    im22 = torch.from_numpy(np.array([[im2]])).type(dt)
    ms_ssim_ = ms_ssim(im11, im22,data_range=im22.max()).data.cpu().numpy()[np.newaxis][0]
    return vif_, ms_ssim_, ssim_, psnr_


def main_return(hparams, num_channels, num_layers, input_size, kernel_size, lr, numit): 

    # Get maksed image from the validation set
    filename = './' + hparams.file_name #'./file1339.h5' 
    f = h5py.File(filename, 'r') 

    # which slice to consider in the following
    slicenu = f["kspace"].shape[0]//2
    slice_ksp = f['kspace'][slicenu] #shape(15, 640, 368)
    slice_ksp_torchtensor = transform.to_tensor(slice_ksp)      # Convert from numpy array to pytorch tensor

    # masking
    try: # if the file already has a mask
        temp = np.array([1 if e else 0 for e in f["mask"]])
        temp = temp[np.newaxis].T
        temp = np.array([[temp]])
        mask = transform.to_tensor(temp).type(dtype).detach().cpu()
    except: # if we need to create a mask
        desired_factor = 4 # desired under-sampling factor
        undersampling_factor = 0
        tolerance = 0.03
        while undersampling_factor < desired_factor - tolerance or undersampling_factor > desired_factor + tolerance:
            mask_func = MaskFunc(center_fractions=[0.07], accelerations=[desired_factor])  # Create the mask function object
            masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor, mask_func=mask_func)   # Apply the mask to k-space
            mask1d = var_to_np(mask)[0,:,0]
            undersampling_factor = len(mask1d) / sum(mask1d)
    mask1d = var_to_np(mask)[0,:,0]
    # The provided mask and data have last dim of 368, but the actual data is smaller. To prevent forcing the network to learn outside the data region, we force the mask to 0 there.
    mask1d[:mask1d.shape[-1]//2-160] = 0 
    mask1d[mask1d.shape[-1]//2+160:] =0
    mask2d = np.repeat(mask1d[None,:], slice_ksp.shape[1], axis=0).astype(int) # Turning 1D Mask into 2D that matches data dimensions
    mask2d = np.pad(mask2d,((0,),((slice_ksp.shape[-1]-mask2d.shape[-1])//2,)),mode='constant') # Zero padding to make sure dimensions match up
    mask = transform.to_tensor( np.array( [[mask2d[0][np.newaxis].T]] ) ).type(dtype).detach().cpu()
    # print('-----------MASKING done')


    ### fit 
    # print('-----------FITTING begins')
    output_depth = slice_ksp.shape[0]*2
    out_size = slice_ksp.shape[1:]
    
    if hparams.arch_name == 'ConvDecoder':
        strides = [1]*(num_layers-1)
        in_size = [input_size*2, input_size]
        net = convdecoder(out_size,
                            in_size,
                            output_depth,
                            num_layers,
                            strides,
                            num_channels, 
                            act_fun = nn.ReLU(),
                            skips=False,
                            need_sigmoid=False,
                            bias=False, 
                            need_last = True,
                            kernel_size=kernel_size,
                            upsample_mode="nearest").type(dtype)
        # print("#prameters of {}:".format(hparams.arch_name),num_param(net))
    
    elif hparams.arch_name == 'DD':
        in_size = [input_size,input_size]
        net = skipdecoder(out_size,
                            in_size,
                            output_depth,
                            num_layers,
                            num_channels,
                            skips=False,
                            need_last=True,
                            need_sigmoid=False,
                            upsample_mode="bilinear").type(dtype)
        # print("#prameters of {}:".format(hparams.arch_name),num_param(net))

    net,ni,slice_ksp_cd = fit_untrained(net, num_channels, num_layers, mask2d, mask, in_size, slice_ksp, slice_ksp_torchtensor, LR=lr, num_iter=numit)
    reconstruction = data_consistency(net, ni, mask1d, slice_ksp_cd)
    # print('-----------FITTING done')

    ### evaluation
    # print('-----------EVALUATION begins')
    gt = f["reconstruction_rss"][slicenu]
    vif_cd, ms_ssim_cd, ssim_cd, psnr_cd  = scores(gt, reconstruction)
    # print('-----------EVALUATION done')


    ### Return result 
    return vif_cd, ms_ssim_cd, ssim_cd, psnr_cd


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
    _, _, _, psnr = main_return(hparams, hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3], hyperparameters[4], num_iters)
    print(f'chn {hyperparameters[0]}, lay {hyperparameters[1]}, ipt {hyperparameters[2]}, fit {hyperparameters[3]}, lrn {hyperparameters[4]}, PSNR {psnr}')
    return psnr 


def successive_halving(hparams, eta=3, max_iter=5000):   #TODO: max_iter 20000?
    T = get_random_hyperparameter_configuration(hparams, ratio=0.1, success_halving=True)
    n = len(T) # initial number of configurations
    # print(f'SUCC length: {n}')
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
    

def greedy(hparams, num_iters=10000):
    chn_ls = [32,64,128,192,256,320,384,448,512]
    lay_ls = [2,4,6,8,10,14,18]#,22,26,32] #OCT20
    ipt_ls = [2,4,8,16,24,32,48,128,256]#,512] #OCT11
    # fit_ls = [4,8,16,32,48,64]
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
    
    # if hparams.fit != fit_ls[0] and hparams.fit != fit_ls[-1]:
    #     anchor = min(fit_ls, key=lambda x:abs(x-hparams.fit))
    #     if hparams.fit < anchor:
    #         fit_ab = [ fit_ls[fit_ls.index(anchor)-1], hparams.fit, anchor ]
    #     if hparams.fit == anchor:
    #         fit_ab = [ fit_ls[fit_ls.index(hparams.fit)-1], hparams.fit, fit_ls[fit_ls.index(hparams.fit)+1] ]
    #     if hparams.fit > anchor:
    #         fit_ab = [ anchor, hparams.fit, fit_ls[fit_ls.index(anchor)+1] ]
    #     configuration_state['fit'] = 1
    # else:
    #     fit_ab = [ hparams.fit, hparams.fit, hparams.fit ]
    #     configuration_state['fit'] = 0
    
    anchor = min(lrn_ls, key=lambda x:abs(x-hparams.lrn))
    lrn_ab = [ anchor, anchor, anchor ]
    configuration_state['lrn'] = 0

    configuration_dic = {'chn':chn_ab, 'lay':lay_ab, 'ipt':ipt_ab, 'lrn':lrn_ab} #configuration_dic = {'chn':chn_ab, 'lay':lay_ab, 'ipt':ipt_ab, 'fit':fit_ab, 'lrn':lrn_ab}

    #Step 2: fine tuning
    def g_return(j, hp_name, config):
        if j == 0:
            value = int((config[hp_name][0] + config[hp_name][1])/2)
        elif j == 2:
            value = int((config[hp_name][1] + config[hp_name][2])/2)
        else:
            value = config[hp_name][1]
        
        if hp_name == 'chn':
            _, _, _, r = main_return(hparams, num_channels=value, num_layers=config['lay'][1], input_size=config['ipt'][1], kernel_size=3, lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'lay':
            _, _, _, r = main_return(hparams, num_channels=config['chn'][1], num_layers=value, input_size=config['ipt'][1], kernel_size=3, lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'ipt':
            _, _, _, r = main_return(hparams, num_channels=config['chn'][1], num_layers=config['lay'][1], input_size=value, kernel_size=3, lr=config['lrn'][1], numit=num_iters)
        # elif hp_name == 'fit':
        #     r = main_return(hparams, num_channels=config['chn'][1], num_layers=config['lay'][1], input_size=config['ipt'][1], kernel_size=value, lr=config['lrn'][1], numit=num_iters)
        elif hp_name == 'lrn':
            _, _, _, r = main_return(hparams, num_channels=config['chn'][1], num_layers=config['lay'][1], input_size=config['ipt'][1], kernel_size=3, lr=value, numit=num_iters)
        
        return r


    # configuration_state = {'chn':0, 'lay':1, 'ipt':1, 'fit':0, 'lrn':0}
    round_count = 0
    while sum(list(configuration_state.values())) != 0:
        print(configuration_state)
        print('prior info of round {}: #chn {}, #lay {}, ipt {}, lrn {}'.format(
                round_count, configuration_dic['chn'][1], configuration_dic['lay'][1], configuration_dic['ipt'][1], configuration_dic['lrn'][1]))
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
        print('result of round {}: #chn {}, #lay {}, ipt {}, lrn {}'.format(
                round_count, configuration_dic['chn'][1], configuration_dic['lay'][1], configuration_dic['ipt'][1], configuration_dic['lrn'][1]))
        #print(configuration_state)
        print('\n')
        round_count += 1
    print('fine tune completed')
    return round_count, configuration_dic

        
if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--pickle_file_path', type=str, default='hyperband_rsl.pkl') #######################################
    PARSER.add_argument('--hyperband_mode', type=int, default=1) #######################################
    PARSER.add_argument('--stype', type=str, default='successive_halving') #######################################
    PARSER.add_argument('--output_file_path', type=str, default='oct14_succe.txt') #######################################
    PARSER.add_argument('--file_name', type=str, default='file1001191.h5')

    # Selection
    PARSER.add_argument('--chn_ls', type=int, nargs='+', default=[32,64,128,192,256,320,384,448,512], help='a list of channels')
    PARSER.add_argument('--lay_ls', type=int, nargs='+', default=[2,4,6,8,10], help='a list of layers')
    PARSER.add_argument('--ipt_ls', type=int, nargs='+', default=[2,4,8,16,24,32,48,128,256], help='a list of input_size')
    PARSER.add_argument('--fit_ls', type=int, nargs='+', default=[3], help='a list of filters') #default=[4,8,16,32,48,64], help='a list of filters')
    PARSER.add_argument('--lr_ls', type=float, nargs='+', default=[0.0005,0.001,0.003,0.005,0.007,0.01,0.002,0.025,0.05], help='a list of learning rate')

    PARSER.add_argument('--chn', type=int, default=32, help='channels')
    PARSER.add_argument('--lay', type=int, default=32, help='layers')
    PARSER.add_argument('--ipt', type=int, default=32, help='input_size')
    PARSER.add_argument('--lrn', type=float, default=0.005, help='learning rate')
    PARSER.add_argument('--arch_name', type=str, default='DD')
    
    HPARAMS = PARSER.parse_args()

    file = sys.stdout   
    sys.stdout = open(HPARAMS.output_file_path, 'w')         
    
    if HPARAMS.stype == 'successive_halving':
        out_conig = successive_halving(HPARAMS)
        #out_conig = hyperband(HPARAMS)
        try:
            if len(out_conig) > 1:
                out_conig = out_conig[0]
            print('Best Config of {}: #channel is {}, #layer is {}, input_size is {}, filter_size is {}, step_size is {}'.format('MRI', out_conig[0][0], 
                out_conig[0][1], out_conig[0][2], out_conig[0][3], out_conig[0][4]))
        except:
            print(f'type, {type(out_conig)}')
            print(f'len, {len(out_conig)}')
            print(f'origin, {out_conig}')
            
    else:
        round_count, final_dict = greedy(HPARAMS)
        print('After {} rounds, fine-tuned Config of {}: #channel is {}, #layer is {}, input_size is {}, step_size is {}'.format(
            round_count, 'MRI', final_dict['chn'][1], final_dict['lay'][1], final_dict['ipt'][1], final_dict['lrn'][1]))







