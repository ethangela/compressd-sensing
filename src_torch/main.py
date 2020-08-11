"""Compressed sensing main script torch"""
from __future__ import print_function
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
import copy

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor

from helpers import *

# == decoder part == #
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

def conv(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)        

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)
    
def set_to(tensor,mtx):
    if not len(tensor.shape)==4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[0]
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if i == j:
                tensor[i,j] = np_to_tensor(mtx)
            else:
                tensor[i,j] = np_to_tensor(np.zeros(mtx.shape))
    return tensor

def conv2(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)

    if kernel_size != 4:
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    else:
        padder = nn.ReflectionPad2d( (1,0,1,0) )
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=1, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver])
    
#model 0 
def decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad, bias=bias))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad,bias=bias))        
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model

#model 1/3: upsample_mode='bilinear'/'none'
def fixed_decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False,
        mtx = np.array( [[1,3,3,1] , [3,9,9,3], [3,9,9,3], [1,3,3,1] ] )*1/16.
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            # those will be fixed
            model.add(conv2( num_channels_up[i], num_channels_up[i],  4, 1, pad=pad))  
            # those will be learned
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            # those will be fixed
            model.add(conv2( num_channels_up[i], num_channels_up[i],  4, 1, pad=pad))  
            # those will be learned
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad))      
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
        
    # set filters to fixed and then set the gradients to zero
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if(m.kernel_size == mtx.shape):
                m.weight.data = set_to(m.weight.data,mtx)
                for param in m.parameters():
                    param.requires_grad = False
    
    return model

#model 2/4: upsample_mode='bilinear'/'none'
def deconv_decoder(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False,
        filter_size=1,
        stride=2,
        padding=0,
        output_padding=0
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add( nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1], filter_size=filter_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias, groups=1, dilation=1) )
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            model.add( nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1], filter_size=filter_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias, groups=1, dilation=1) )
    
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model




# == fit part == #
def sqnorm(a):
    return np.sum( a*a )

def get_distances(initial_maps,final_maps):
    results = []
    for a,b in zip(initial_maps,final_maps):
        res = sqnorm(a-b)/(sqnorm(a) + sqnorm(b))
        results += [res]
    return(results)

def get_weights(net):
    weights = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            weights += [m.weight.data.cpu().numpy()]
    return weights

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def fit(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter = 10000,
        LR = 0.01,
        OPTIMIZER='adam',
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask_var = None,
        apply_f = None,
        lr_decay_epoch = 0,
        net_input = None,
        net_input_gen = "random",
        find_best=False,
        weight_decay=0,
        upsample_mode = "bilinear",
        totalupsample = 1,
        loss_type="MSE",
        output_gradients=False,
        output_weights=False,
        show_images=False,
       ):
    
    # build input noise 
    if net_input is not None:
        print("input provided")
    else:
        if upsample_mode=="bilinear":
            # feed uniform noise into the network 
            totalupsample = 2**len(num_channels)
        elif upsample_mode=="deconv":
            # feed uniform noise into the network 
            totalupsample = 2**(len(num_channels)-1)
        #############################################################
        #width = int(img_clean_var.data.shape[2]/totalupsample)
        #height = int(img_clean_var.data.shape[3]/totalupsample)
        width = int(img_clean_var.data.shape[2])
        height = int(img_clean_var.data.shape[3])
        #############################################################
        shape = [1,num_channels[0], width, height]
        net_input = Variable(torch.zeros(shape)).type(dtype)
        net_input.data.uniform_()
        net_input.data *= 1./10
    
    # restore the (constant)noise built
    net_input = net_input.type(dtype)
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters()]
    if(opt_input == True): # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

    # build loss list
    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)
    
    # optimiser
    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    elif OPTIMIZER == 'LBFGS':
        print("optimize with LBFGS", LR)
        optimizer = torch.optim.LBFGS(p, lr=LR)

    if loss_type=="MSE":
        mse = torch.nn.MSELoss() #.type(dtype) 
    if loss_type=="L1":
        mse = nn.L1Loss()
    
    # best result saver 
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    # gradients record matrix
    nconvnets = 0
    for p in list(filter(lambda p: len(p.data.shape)>2, net.parameters())):
        nconvnets += 1
    out_grads = np.zeros((nconvnets,num_iter))
    
    # weights(conv) record matrix 
    init_weights = get_weights(net)
    out_weights = np.zeros((len(init_weights), num_iter))
    
    # convolutions 
    for i in range(num_iter):
        
        # learning rate decay setting
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)
        
        # optional additional noise to the constant input 
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        
        # for each step:
        def closure():
            
            # initialisation 
            optimizer.zero_grad()
            
            # deep decoder output 
            out = net(net_input.type(dtype))
            
            # training loss 
            if mask_var is not None:
                loss = mse( out * mask_var , img_noisy_var * mask_var )
            elif apply_f: #loss in compressed sensing 
                loss = mse(apply_f(out) , img_noisy_var )
            else: #loss without compressed sensing 
                loss = mse(out, img_noisy_var)
        
            # pytorch necessary command to process backpropogation 
            loss.backward()
            
            # record compressed sensing loss list on cpu
            mse_wrt_noisy[i] = loss.data.cpu().numpy()

            # record loss list (without compressed sensing) on cpu   
            true_loss = mse(Variable(out.data, requires_grad=False).type(dtype), img_clean_var.type(dtype))
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()
            
            # record gradient  
            if output_gradients:
                for ind,p in enumerate(list(filter(lambda p: p.grad is not None and len(p.data.shape)>2, net.parameters()))):
                    out_grads[ind,i] = p.grad.data.norm(2).item()
            
            # output results during iterations 
            if i % 10 == 0:
                out2 = net(Variable(net_input_saved).type(dtype))
                loss2 = mse(out2, img_clean_var)
                print ('Iteration %05d  Train loss %f  Actual loss %f Actual loss orig %f' % (i, loss.data,true_loss.data,loss2.data), '\r', end='')
            
            # show images 
            if show_images:
                if i % 50 == 0:
                    print(i)
                    out_img_np = net( ni.type(dtype) ).data.cpu().numpy()[0]
                    myimgshow(plt,out_img_np)
                    plt.show()
            
            # record weights
            if output_weights:
                out_weights[:,i] = np.array( get_distances( init_weights, get_weights(net) ) )
            
            # final return 
            return loss   
        
        # pytorch necessary command to process optimisation 
        loss = optimizer.step(closure)
            
        # best result saver
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net = copy.deepcopy(net)
                    
    if find_best:
        net = best_net
    if output_gradients and output_weights:
        return mse_wrt_noisy, mse_wrt_truth, net_input_saved, net, out_grads
    elif output_gradients:
        return mse_wrt_noisy, mse_wrt_truth, net_input_saved, net, out_grads      
    elif output_weights:
        return mse_wrt_noisy, mse_wrt_truth, net_input_saved, net, out_weights
    else:
        return mse_wrt_noisy, mse_wrt_truth, net_input_saved, net 




# == load input == #
def load_2D(path, img_name):
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = img[None, :, :, :]
    img_ts = torch.from_numpy(img)    
    img_var = Variable(img_ts)
    img_var = img_var.type(dtype)
    return img_var
    
def load_img(path, img_name):
    img_path = os.path.join(path, img_name)
    img = imread(img_path)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = img[None, :, :, :]
    img_clean = img / 255.
    img_ts = torch.from_numpy(img_clean)    
    img_var = Variable(img_ts)
    img_var = img_var.type(dtype)
    return img_var

img_np = load_2D('', '2D_rbf_2.npy')
img_ts = torch.from_numpy(img_np)    
img_var = Variable(img_ts)
img_var = img_var.type(dtype) #[1,1,64,64]




# == compressed sensing == #
def dd_recovery(input_net, img_var, num_channels, output_channel, num_iter=10000, ni=None):
    
    # define measurement
    X = img_var.view(-1, np.prod(img_var.shape) ) #[1,4096]
    n = X.shape[-1]
    m = int(n/3)
    A = torch.empty(n,m).uniform_(0, 1).type(dtype)
    A *= 1/np.sqrt(m)
    def forwardm(img_var):
        X = img_var.view(-1, np.prod(img_var.shape))
        return torch.mm(X,A)
    measurement = forwardm(img_var)
    
    # define the net 
    net = input_net(num_output_channels=output_channel, num_channels_up=num_channels).type(dtype)
    
    # fitting 
    mse_n, mse_t, ni, net = fit(net=net,
                                img_noisy_var=measurement.type(dtype),
                                num_channels=num_channels,
                                net_input=ni,
                                reg_noise_std=0.0,
                                num_iter=num_iter,
                                LR = 0.005,
                                apply_f = forwardm,
                                img_clean_var=img_var.type(dtype),
                                upsample_mode='bilinear'
                                )
    
    # applyed "trained" net on input noise 
    out_img_var = net(ni.type(dtype))
    
    return mse_n, mse_t, out_img_var
    
    

    
# == main part == #
def main(hparams):
    #parameters 
    k = hparams.k
    num_channels = [k]*hparams.num_channel
    output_channel = hparams.output_channel
    
    #input image 
    if hparams.image_mode == '2D':
        img_var = load_2D(hparams.path, hparams.name)
    elif hparams.image_mode == '3D':
        img_var = load_img(hparams.path, hparams.name)
    
    #deep decoder 
    if hparams.decoder_type == 'original':
        input_net = decodernw()
    elif hparams.decoder_type == 'fixed_deconv':
        input_net = fixed_decodernw()
    elif hparams.decoder_type == 'deconv':
        input_net = deconv_decoder()
    
    mse_n, mse_t, out_img_var = dd_recovery(input_net, img_var, num_channels, output_channel)
    torch.save(out_img_var, 'output_image.pt')
    print('final measurement loss: {}'.format(mse_n))
    print('final recovery loss: {}'.format(mse_t))
    
    
    
    
if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--path', type=str, default='', help='path stroing the images')
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images') ###################################
    PARSER.add_argument('--img_name', type=str, default='1D_rbf_2.npy', help='image to use') ###################################

    # Deep decoder 
    PARSER.add_argument('--k', type=int, default=64, help='number of channel dimension')
    PARSER.add_argument('--num_channel', type=int, default=4, help='number of upsample channles')
    PARSER.add_argument('--decoder_type', type=str, default='fixed_deconv', help='decoder type') ###################################
 
    HPARAMS = PARSER.parse_args()
    
    main(HPARAMS)
