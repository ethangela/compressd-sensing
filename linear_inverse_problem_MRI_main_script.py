from __future__ import print_function
# import matplotlib.pyplot as plt
#%matplotlib notebook

import os
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
import pandas as pd
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




def fit_untrained(parnet, num_channels, num_layers, mask2d_, mask, in_size, slice_ksp, slice_ksp_torchtensor, LR, num_iter=20000):
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


def main_return(hparams, num_channels, num_layers, input_size, kernel_size, lr, numit, arch_name='DD'): 

    ### Get maksed image from the validation set
    print('-----------LOADING and MASKING begin')
    # proxy_handler = urll.ProxyHandler({})
    # opener = urll.build_opener(proxy_handler)
    # urll.install_opener(opener)
    # data_url =  'https://rice.box.com/shared/static/y1tcaa0eo62ie3lszrkamqivdvirx1x3.h5'
    # req = urll.Request(data_url)
    # r = opener.open(req)
    # result = r.read()
    filename = './' + hparams.file_name #'./file1339.h5' 
    # with open(filename, 'wb') as f:
    #     f.write(result)
    f = h5py.File(filename, 'r') 
    print("Kspace shape (number slices, number coils, x, y): ", f['kspace'].shape)

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
    print("under-sampling factor:",round(len(mask1d)/sum(mask1d),2))
    print('-----------MASKING done')


    ### fit 
    print('-----------FITTING begins')
    output_depth = slice_ksp.shape[0]*2
    out_size = slice_ksp.shape[1:]
    
    if arch_name == 'ConvDecoder':
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
        print("#prameters of {}:".format(arch_name),num_param(net))
    
    elif arch_name == 'DD':
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
        print("#prameters of {}:".format(arch_name),num_param(net))

    elif arch_name == "DIP":
        in_size = slice_ksp.shape[-2:]
        pad = "zero" #'reflection' # 'zero'
        net = skip(in_size,
                    num_channels, 
                    output_depth, 
                    num_channels_down = [num_channels] * 8,
                    num_channels_up =   [num_channels] * 8,
                    num_channels_skip = [num_channels*0] * 6 + [4,4],  
                    filter_size_up = 3, 
                    filter_size_down = 5, 
                    upsample_mode='nearest', 
                    filter_skip_size=1,
                    need_sigmoid=False, 
                    need_bias=True, 
                    pad=pad, 
                    act_fun='ReLU').type(dtype)
        print("#prameters of {}:".format(arch_name),num_param(net))
        

    net,ni,slice_ksp_cd = fit_untrained(net, num_channels, num_layers, mask2d, mask, in_size, slice_ksp, slice_ksp_torchtensor, LR=lr, num_iter=numit)
    reconstruction = data_consistency(net, ni, mask1d, slice_ksp_cd)
    print('-----------FITTING done')

    ### evaluation
    print('-----------EVALUATION begins')
    gt = f["reconstruction_rss"][slicenu]
    vif_cd, ms_ssim_cd, ssim_cd, psnr_cd  = scores(gt, reconstruction)
    print('-----------EVALUATION done')


    ### Return result 
    return vif_cd, ms_ssim_cd, ssim_cd, psnr_cd, reconstruction, gt



if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--pickle_file_path', type=str, default='MRI_test.pkl') #######################################
    PARSER.add_argument('--stype', type=str, default='successive_halving') #######################################
    PARSER.add_argument('--output_file_path', type=str, default='oct14_succe.txt') #######################################
    PARSER.add_argument('--file_name', type=str, default='file1001191.h5')
    PARSER.add_argument('--opt_mode', type=str, default='optimized')

    # Selection
    PARSER.add_argument('--chn_ls', type=int, nargs='+', default=[32,64,128,192,256,320,384,448,512], help='a list of channels')
    PARSER.add_argument('--lay_ls', type=int, nargs='+', default=[2,4,6,8,10,14,18,22,26,32], help='a list of layers')
    PARSER.add_argument('--ipt_ls', type=int, nargs='+', default=[2,4,8,16,24,32,48,128,256], help='a list of input_size')
    PARSER.add_argument('--fit_ls', type=int, nargs='+', default=[3], help='a list of filters') #default=[4,8,16,32,48,64], help='a list of filters')
    PARSER.add_argument('--lr_ls', type=float, nargs='+', default=[0.0005,0.001,0.003,0.005,0.007,0.01,0.002,0.025,0.05], help='a list of learning rate')

    PARSER.add_argument('--chn', type=int, default=32, help='channels')
    PARSER.add_argument('--lay', type=int, default=32, help='layers')
    PARSER.add_argument('--ipt', type=int, default=32, help='input_size')
    PARSER.add_argument('--fit', type=int, default=3, help='filters')
    PARSER.add_argument('--lrn', type=float, default=0.005, help='learning rate')
    PARSER.add_argument('--arch_name', type=str, default='DD')
    
    HPARAMS = PARSER.parse_args()

    file = sys.stdout   
    sys.stdout = open(HPARAMS.output_file_path, 'w')    

    # return
    vif_cd, ms_ssim_cd, ssim_cd, psnr_cd, reconstruction, gt = main_return(HPARAMS, 
        num_channels=HPARAMS.chn, num_layers=HPARAMS.lay, input_size=HPARAMS.ipt, kernel_size=HPARAMS.fit, lr=HPARAMS.lrn, numit=20000, arch_name=HPARAMS.arch_name)

    # table
    if not os.path.exists(HPARAMS.pickle_file_path):
        d = {'file':[HPARAMS.file_name], 'optimization_mode':[HPARAMS.opt_mode], 'channel':[HPARAMS.chn], 'layer':[HPARAMS.lay], 'input_size':[HPARAMS.ipt], 
            'filter_size':[HPARAMS.fit], 'step_size':[HPARAMS.lrn], 'VIF':[vif_cd], 'MS-SSIM':[ms_ssim_cd], 'SSIM':[ssim_cd], 'PSNR':[psnr_cd]}
        df = pd.DataFrame(data=d)
        df.to_pickle(HPARAMS.pickle_file_path)
    else:
        d = {'file':HPARAMS.file_name, 'optimization_mode':HPARAMS.opt_mode, 'channel':HPARAMS.chn, 'layer':HPARAMS.lay, 'input_size':HPARAMS.ipt, 
            'filter_size':HPARAMS.fit, 'step_size':HPARAMS.lrn, 'VIF':vif_cd, 'MS-SSIM':ms_ssim_cd, 'SSIM':ssim_cd, 'PSNR':psnr_cd}
        df = pd.read_pickle(HPARAMS.pickle_file_path)
        df = df.append(d, ignore_index=True)
        df.to_pickle(HPARAMS.pickle_file_path)

    # plot
    img_name = f'{HPARAMS.file_name[:-3]}_{HPARAMS.opt_mode}_reconstruct'
    plt.figure(figsize=(6,6))
    plt.imshow(reconstruction, "gray")
    plt.axis("off")
    plt.savefig(img_name+'.png')

    img_name = f'{HPARAMS.file_name[:-3]}_original'
    plt.figure(figsize=(6,6))
    plt.imshow(gt, "gray")
    plt.axis("off")
    plt.savefig(img_name+'.png')
