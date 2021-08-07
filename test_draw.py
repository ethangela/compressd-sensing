import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from skimage.io import imread, imsave
#import torch.nn as nn 
#import torch.optim as optim
import random
import pywt
import math

def kernel(x1, x2, l=0.5, sigma_f=0.2):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

def sample_from_2D_Gaussian(size):
    m = size**2
    test_d1 = np.linspace(-10, 10, m)
    test_d2 = np.linspace(-10, 10, m)
    
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]
    test_X = np.asarray(test_X)
    mu = np.zeros_like(test_d1)
    cov = kernel(test_X, test_X)
    print('parameter set done')
    
    gp_samples = np.random.multivariate_normal(
            mean = mu.ravel(), 
            cov = cov, 
            size = 1)
    z = gp_samples.reshape(test_d1.shape)
    print('sampling done')
    
    #scale to range(0,1)
    z = (z - np.min(z))/np.ptp(z)
    np.save('2D.npy', z)
    #print(z)
    test_d1 = (test_d1 - np.min(test_d1))/np.ptp(test_d1)
    test_d2 = (test_d2 - np.min(test_d2))/np.ptp(test_d2)
    print('scaling done')
    
    fig = plt.figure(figsize=(5, 5))
    plt.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=1)
    #ax.set_title("with optimization l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    #plt.show()
    plt.savefig('Jun25.jpg')
    
def excel(path):
    df = pd.read_pickle(path)
    csv_title = path[:-3] + 'csv'
    df.to_csv(csv_title,index=0)
    pass








def plot_lay(lay_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))
 
    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 

    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['channels'] == 256) & (df['input_size'] == 768) & (df['filter_size'] == 4)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='layers')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(lay_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr)
    
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(lay_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('layers')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('layers for {} on signal {}'.format(mask_inf, type))
    # show a legend on the plot
    plt.legend()

    path_dir = 'figures/layer/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/layer_for_{}_on_{}_signal.png'.format(mask_inf, type))

def plot_chn(chn_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path) 
    plt.figure(figsize=(16,10))
    
    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy']    

    if mask_inf != 'circulant':
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['filter_size'] == 15)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='channels')
    y_avg = []
    for sgl in sgl_list:
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        plt.plot(chn_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr)
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(chn_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    
    
    plt.xlabel('channels')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('channels for {} on signal {}'.format(mask_inf, type))
    # show a legend on the plot
    plt.legend()
    
    path_dir = 'figures/channel/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/channel_for_{}_on_{}_signal.png'.format(mask_inf, type))




def plot_ipt(ipt_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy']#, '1D_rbf_3.0_4.npy', '1D_rbf_3.0_5.npy'] ##########

    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['channels'] == 6) & (df['filter_size'] == 15)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='input_size')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(ipt_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(ipt_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('input_sizes')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('input_size for {} on signal {}'.format(mask_inf, type))    
    # show a legend on the plot
    plt.legend()

    path_dir = 'figures/input_size/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/input_size_for_{}_on_{}_signal.png'.format(mask_inf, type))




def plot_fit(type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 
    
    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['channels'] == 6) & (df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['channels'] == 6  )] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='filter_size')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        if len(psnr) > 10:
            p = []
            l = int(len(psnr)/3)
            for i in range(3):
                p.append(psnr[i*l:(i+1)*l])
            psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(fit_size, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(fit_size, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('filter_sizes')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('filter_size for {} on signal {}'.format(mask_inf, type)) 
    # show a legend on the plot
    plt.legend()
    path_dir = 'figures/single_layer/filter_size/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass 
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/filter_size_for_{}_on_{}_signal.png'.format(mask_inf, type))





def plot_activation(type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 
    
    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['channels'] == 384) & (df['layers'] == 4) & (df['input_size'] == 384) & (df['filter_size'] == 10) & (df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['channels'] == 384) & (df['layers'] == 4) & (df['input_size'] == 384) & (df['filter_size'] == 10)] #a set of channels on 3 signals THREE TIMES?#########

    act_list = ['relu', 'leaky_relu', 'sigmoid']
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        plt.plot(act_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(act_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('act_functions')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('act_function for {} on signal {}'.format(mask_inf, type)) 
    # show a legend on the plot
    plt.legend()
    path_dir = 'figures/single_layer/act_function/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass 
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/act_function{}_on_{}_signal.png'.format(mask_inf, type))



def picture(path, mask=None, type='block'):
    plt.figure(figsize=(80,10))
    xs = np.linspace(-10,10,4096) #Range vector (101,)
    xs = (xs - np.min(xs))/np.ptp(xs)
    fs = np.load(path)
    print('signal',fs.shape)
    title = path[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    
    if mask:
        m = np.load('mask/1D_mask_'+type+'_4096_'+str(mask)+'_1.npy')
        fs = fs * m
        fs[fs == 0.] = np.nan
        title += '_masked_'+type+'_'+str(mask)
    
    plt.plot(xs, fs, 'gray') # Plot the samples
    plt.savefig(tn + '.jpg')
    plt.close()


def group_pic(org, msk, path, title_m):
    org1, org2, org3 = org
    path1, path2, path3 = path

    plt.figure(figsize=(240,30))
    
    xs = np.linspace(-10,10,4096) #Range vector (101,)
    xs = (xs - np.min(xs))/np.ptp(xs)

    # original
    plt.subplot(331)
    fs = np.load(org1)
    plt.title(org1[:-4])
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(334)
    fs = np.load(org2)
    plt.title(org2[:-4])
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(337)
    fs = np.load(org3)
    plt.title(org3[:-4])
    plt.plot(xs, fs, 'gray')


    # masked
    plt.subplot(332)
    m = np.load(msk)
    fs = np.load(org1)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(335)
    m = np.load(msk)
    fs = np.load(org2)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(338)
    m = np.load(msk)
    fs = np.load(org3)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')


    # recovered
    plt.subplot(333)
    fs = np.load(path1)
    title = path1[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.subplot(336)
    fs = np.load(path2)
    title = path2[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.subplot(339)
    fs = np.load(path3)
    title = path3[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.savefig(title_m + '.jpg')
    print('saved')
    plt.close()


def ipt(hparams):
    o = 4096 / (pow(hparams.f, hparams.l))
    print(o)




from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 15 })
import img2pdf

def plot_jan(csv_path, mea_type, mea_range, sig_type, mask_inf):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,5))
 
    if sig_type == 'exp':
        sgl_list = ['1D_exp_0.25_4096_10.npy', '1D_exp_0.25_4096_11.npy', '1D_exp_0.25_4096_12.npy', '1D_exp_0.25_4096_13.npy', '1D_exp_0.25_4096_14.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_4096_10.npy', '1D_rbf_3.0_4096_11.npy', '1D_rbf_3.0_4096_12.npy', '1D_rbf_3.0_4096_13.npy', '1D_rbf_3.0_4096_14.npy'] 

    df_sel = df

    # if mea_type == 'layers':
    #     #lay_list = [1,2,3,4,5,6,7,8,9,10,12,16,20,25,30,35,40,50,60,70,85,100,120,140,180,240] #3,3,7
    # elif mea_type == 'channels':
    #     #lay_list = [50,80,120,140,160,180,200,220,260,300,350,400,450,500,600,700,850,1000]#160,180,160
    # elif mea_type == 'input_size':
    #     #lay_list = [30,50,70,80,90,95,100,105,110,120,130,140,150,180,200,240,330,384,420,480,520]#120,200,384

    lay_list = mea_range

    df_sel = df_sel.sort_values(by = mea_type)##### e.g.'layers'
    y_avg = []
    for i,sgl in enumerate(sgl_list):
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        label_title = 'testing signal {}'.format(int(i+1))
        plt.plot(lay_list, psnr, '--', label = label_title, alpha=0.6)#####
        y_avg.append(psnr)
    
    y_plt = [(x+y+z+e+f)/5 for x,y,z,e,f in zip(*y_avg)]
    plt.plot(lay_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel(mea_type)#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    mask_info_ls = mask_inf.split('_') #'random_8', 'block_2', 'denoise_0.05', 'compress_100'
    if mask_info_ls[0] == 'random' or mask_info_ls[0] == 'block':
        mask_info = '1_{}_of_entries_are_{}-masked'.format(int(mask_info_ls[1]), mask_info_ls[0])
    elif mask_info_ls[0] == 'denoise':
        mask_info = 'medium_noise_added'
    elif mask_info_ls[0] == 'compress':
        mask_info = 'compressing_at_ratio_of_{}_4096'.format(int(mask_info_ls[1]))
    plot_title =  mea_type + '_for_'+ sig_type +'_signals_with_measurement_as_'+ mask_info
    #plt.title(plot_title)
    # show a legend on the plot
    plt.legend()

    path_dir = '1ECML/arch/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    img_name = plot_title +'.jpg'
    pdf_name = plot_title +'.pdf'
    final_img = os.path.join(path_dir, img_name)
    final_pdf = os.path.join(path_dir, pdf_name)
    plt.savefig(final_img)
    pdf_bytes = img2pdf.convert(final_img)
    file_ = open(final_pdf, "wb")
    file_.write(pdf_bytes)
    file_.close()


def qplot(sig_type, sig='exp_0.25', dec='lasso_wavelet', mea='denoise', msk='block_4096_3_4'): 
    plt.figure(figsize=(13,3))
    ax = plt.axes(frameon=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    xs = np.linspace(0,4096,4096) 
    # xs = (xs - np.min(xs))/np.ptp(xs)
    if sig_type == 'exp':
        path = 'Gaussian_signal/1D_exp_0.25_4096_29.npy'
        plot_title = 'exp_original'
    elif sig_type == 'rbf':
        path = 'Gaussian_signal/1D_rbf_1.0_4096_29.npy'
        plot_title = 'rbf_original'
    elif sig_type == 'no2':
        xs = np.linspace(0,1024,1024) 
        path = 'Ieeg_signal/NO2_1024_9.npy'
        plot_title = 'no2_original'
    elif sig_type == 'co':
        xs = np.linspace(0,1024,1024) 
        path = 'Ieeg_signal/CO_1024_9.npy'
        plot_title = 'co_original'
    elif sig_type == 'recover':
        if mea == 'denoise':
            path = 'final_recons/1D_{}_4096_30_decoder_type_{}_model_type_denoising_noise_005.npy'.format(sig, dec)
            if dec.split('_')[0] != 'fixed':
                plot_title = '{}_{}_{}_recover'.format(dec.split('_')[0], mea, sig.split('_')[0])
            else:
                plot_title = '{}_{}_recover'.format(mea, sig.split('_')[0])
        elif mea == 'inpaint':
            path = 'final_recons/1D_{}_4096_30_decoder_type_{}_model_type_inpainting_mask_{}_1.npy'.format(sig, dec, msk)
            if dec.split('_')[0] != 'fixed':
                plot_title = '{}_{}_{}_{}_recover'.format(dec.split('_')[0], msk.split('_')[0], msk.split('_')[2], sig.split('_')[0])
            else:
                plot_title = '{}_{}_{}_recover'.format(msk.split('_')[0], msk.split('_')[2], sig.split('_')[0])
    fs = np.load(path)
    fs[fs == 0.] = np.nan
    plt.plot(xs, fs, 'gray')
    # ax.set_xticks([])
    plt.savefig('final_recons/{}.jpg'.format(plot_title))
    pdf_bytes = img2pdf.convert('final_recons/{}.jpg'.format(plot_title))
    file_ = open('final_recons/{}.pdf'.format(plot_title), "wb")
    file_.write(pdf_bytes)
    file_.close()
    print('single plot saved')
    plt.close()

def oplot(sig_type, mea_path, plot_title, mode='1D'): 
    if mode=='1D':
        plt.figure(figsize=(14,4))
        xs = np.linspace(-10,10,4096) 
        xs = (xs - np.min(xs))/np.ptp(xs)
        if sig_type == 'exp':
            path = 'Gaussian_signal/1D_exp_0.25_4096_30.npy'
        elif sig_type == 'rbf':
            path = 'Gaussian_signal/1D_rbf_3.0_4096_30.npy'
        fs = np.load(path)
        if mea_path != 'denoise':
            mea_path = 'Masks/' + mea_path + '.npy'
            ms = np.load(mea_path)
            ms = ms.reshape(4096)
            ys = fs * ms
            ys[ys == 0.] = np.nan
        else:
            ys = fs + 0.05 * np.random.randn(4096)
        plt.plot(xs, ys, 'gray')
        plt.savefig('final_recons/{}.jpg'.format(plot_title))
        print('single plot saved')
        plt.close()
    else:
        path = sig_type +'.jpg'
        def load_img(img_name):
            img_path = os.path.join('Celeb_signal', img_name)
            img = imread(img_path)
            img = np.transpose(img, (1, 0, 2))
            return img
        fs = load_img(path)
        if mea_path != 'denoise':
            mea_path = 'Masks/' + mea_path + '.npy'
            ms = np.load(mea_path)
            ms = ms[:, :, None]
            ys = fs * ms
            ys[ys == 0.] = np.nan
        else:
            ys = fs + 0.05 * np.random.randn(fs.shape[0], fs.shape[1], fs.shape[2])
        ys = np.transpose(ys, (1, 0, 2))
        imsave('2D_recons/{}.jpg'.format(plot_title), ys.astype(np.uint8))

        
    
def iplot(sig_type='exp', compress_ratio=None, mask_info=None, noise_level=None, decoder=None, crop1=None, crop2=None):
    #plt setting
    plt.figure(figsize=(13,3))
    ax = plt.axes(frameon=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    xs = np.linspace(0,crop2-crop1,crop2-crop1)

    #org signal
    if sig_type == 'exp':
        org_path = 'Gaussian_signal/1D_exp_0.25_4096_30.npy'
        sig_detail = '1D_exp_0.25_4096_30'
    elif sig_type == 'rbf':
        org_path = 'Gaussian_signal/1D_rbf_1.0_4096_30.npy'
        sig_detail = '1D_rbf_1.0_4096_30'
    elif sig_type == 'no2':
        org_path = 'Ieeg_signal/NO2_1024_9.npy'
        sig_detail = 'NO2_1024_9'
    elif sig_type == 'co': 
        org_path = 'Ieeg_signal/CO_1024_9.npy'
        sig_detail = 'CO_1024_9'
    org_fs = np.load(org_path)[crop1:crop2]
    org_fs[org_fs == 0.] = np.nan
    plt.plot(xs, org_fs, 'r--', label = 'Signal')#####

    #recovered
    if decoder: 
        label_dic = {'original':'Org.DD', 'deep_image':'Org.DIP', 'tv_norm':'TV Norm', 'lasso_wavelet':'LassoW'}
        if mask_info != '0':
            curve='final_recons/{}_decoder_type_{}_model_type_inpainting_mask_{}.npy'.format(sig_detail, 'fixed_deconv', mask_info)
            curve_='final_recons/{}_decoder_type_{}_model_type_inpainting_mask_{}.npy'.format(sig_detail, decoder, mask_info)
            title_info = mask_info
        elif compress_ratio != '0':
            curve='final_recons/{}_decoder_type_{}_model_type_compressing_num_mea_{}.npy'.format(sig_detail, 'fixed_deconv', compress_ratio)
            curve_='final_recons/{}_decoder_type_{}_model_type_compressing_num_mea_{}.npy'.format(sig_detail, decoder, compress_ratio)
            title_info = 'compress_'+compress_ratio
        elif noise_level != '0':
            curve='final_recons/{}_decoder_type_{}_model_type_denoising_noise_{}.npy'.format(sig_detail, 'fixed_deconv', noise_level)
            curve_='final_recons/{}_decoder_type_{}_model_type_denoising_noise_{}.npy'.format(sig_detail, decoder, noise_level)
            title_info = 'noise_'+noise_level

        rec_crv = np.load(curve)[crop1:crop2]
        rec_crv[rec_crv == 0.] = np.nan
        plt.plot(xs, rec_crv, linestyle='dashdot', label='Opt.DD')

        rec_crv_ = np.load(curve_)[crop1:crop2]
        rec_crv_[rec_crv_ == 0.] = np.nan
        plt.plot(xs, rec_crv_, linestyle='dotted', label=label_dic[decoder])
        
        plt.legend()
        plt.savefig('final_recons/{}_{}_reconstruction.jpg'.format(sig_type,title_info))
        pdf_bytes = img2pdf.convert('final_recons/{}_{}_reconstruction.jpg'.format(sig_type,title_info))
        file_ = open('final_recons/{}_{}_reconstruction.pdf'.format(sig_type,title_info), "wb")
        file_.write(pdf_bytes)
        file_.close()
        print('single plot saved')
        plt.close()

    else:
        if mask_info != '0':
            curve = 'final_recons/{}_decoder_type_{}_model_type_inpainting_mask_{}.npy'.format(sig_detail, 'fixed_deconv', mask_info)
            title_info = mask_info
        elif compress_ratio != '0':
            curve = 'final_recons/{}_decoder_type_{}_model_type_compressing_num_mea_{}.npy'.format(sig_detail, 'fixed_deconv', compress_ratio)
            title_info = 'compress_'+compress_ratio
        elif noise_level != '0':
            curve = 'final_recons/{}_decoder_type_{}_model_type_denoising_noise_{}.npy'.format(sig_detail, 'fixed_deconv', noise_level)
            title_info = 'noise_'+noise_level
        rec_crv = np.load(curve)[crop1:crop2]
        rec_crv[rec_crv == 0.] = np.nan
        plt.plot(xs, rec_crv, linestyle='dashdot', label='Opt.DD')#, alpha=0.6)

        plt.legend()
        plt.savefig('final_recons/{}_{}_reconstruction.jpg'.format(sig_type,title_info))
        pdf_bytes = img2pdf.convert('final_recons/{}_{}_reconstruction.jpg'.format(sig_type,title_info))
        file_ = open('final_recons/{}_{}_reconstruction.pdf'.format(sig_type,title_info), "wb")
        file_.write(pdf_bytes)
        file_.close()
        print('single plot saved')
        plt.close()


    




def main(hparams):
    if hparams.plot == 'arch':
        plot_jan(hparams.csv, 
        hparams.mea_type, 
        hparams.mea_range, 
        hparams.sig_type,
        hparams.mask_inf)
    elif hparams.plot == 'original':
        qplot(hparams.sig_type)
    elif hparams.plot == 'observe':
        oplot(hparams.sig_type,
        hparams.mea_path,
        hparams.title,
        hparams.mode)
    elif hparams.plot == 'recover':
        qplot(hparams.sig_type,
        hparams.sig,
        hparams.dec,
        hparams.mea,
        hparams.msk)
    elif hparams.plot == 'integrate':
        iplot(hparams.sig_type,
        hparams.compress_ratio,
        hparams.mask_info,
        hparams.noise_level,
        hparams.decoder,
        hparams.crop1,
        hparams.crop2)

    print('ploting completed')



if __name__ == '__main__':    
    # PARSER = ArgumentParser()
    # PARSER.add_argument('--csv', type=str, default='inpaint_chn_exp_random_18.pkl', help='path stroing pkl')
    # PARSER.add_argument('--mea_type', type=str, default='channels', help='layers/channels/input_size/filter_size/step_size')
    # PARSER.add_argument('--mea_range', type=float, nargs='+', default=[1,2,3,4,5,6,7,8,9,10,12,16,20], help='a list of xs')
    # PARSER.add_argument('--sig_type', type=str, default='co', help='exp/rbf')
    # PARSER.add_argument('--mask_inf', type=str, default='random_8', help='mask info, e.g., block_2, denoise_0.05, circulant_100') 
    # PARSER.add_argument('--plot', type=str, default='original') 
    # PARSER.add_argument('--mea_path', type=str, default='Masks/1D_mask_random_64_2_1.npy') 
    # PARSER.add_argument('--title', type=str, default='original_rbf') 
    # PARSER.add_argument('--recover_path', type=str, default='recover_path') 
    # PARSER.add_argument('--sig', type=str, default='exp_0.25') 
    # PARSER.add_argument('--dec', type=str, default='lasso_wavelet') 
    # PARSER.add_argument('--mea', type=str, default='denoise') 
    # PARSER.add_argument('--msk', type=str, default='block_4096_3_4') 
    # PARSER.add_argument('--mode', type=str, default='1D') 
    # PARSER.add_argument('--compress_ratio', type=str, default='0')
    # PARSER.add_argument('--mask_info', type=str, default='0')
    # PARSER.add_argument('--noise_level', type=str, default='0')
    # PARSER.add_argument('--decoder', type=str, default=None)
    # PARSER.add_argument('--crop1', type=int, default=0)
    # PARSER.add_argument('--crop2', type=int, default=4096)

    
    # HPARAMS = PARSER.parse_args()
    
    # main(HPARAMS)

    def qplot(sig_type): 
        plt.figure(figsize=(100,5))
        ax = plt.axes(frameon=True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if sig_type == 'no2':
            path = 'Ieeg_signal/NO2.npy'
            fs = np.load(path)
            length = int(fs.shape[0])
            xs = np.linspace(0,length,length) 
            plot_title = 'NO2_original'
        elif sig_type == 'co':
            path = 'Ieeg_signal/CO.npy'
            fs = np.load(path)
            length = int(fs.shape[0])
            xs = np.linspace(0,length,length) 
            plot_title = 'CO_original'
        elif sig_type == 'o3':
            path = 'Ieeg_signal/O3.npy'
            fs = np.load(path)
            length = int(fs.shape[0])
            xs = np.linspace(0,length,length) 
            plot_title = 'O3_original'
        ns = fs.copy()
        x_tick = []
        for i in range(len(ns)-1):
            if ns[i] != 0 and ns[i+1] == 0:
                x_tick.append(i)
            if ns[i] == 0 and ns[i+1] != 0:
                x_tick.append(i)
        ns[ns != 0.] = np.nan
        ns[ns == 0.] = 1
        plt.plot(xs, ns, 'red')
        fs[fs == 0.] = np.nan
        plt.plot(xs, fs, 'gray')
        plt.xticks(np.array(x_tick), fontsize=10)
        plt.xticks(rotation=90)
        plt.title(plot_title)
        plt.savefig('jul9/{}.jpg'.format(plot_title))
        print('single plot saved')
        plt.close()

    # qplot('no2')
    # qplot('co')
    # qplot('o3')

    
    def tailor(sig_type='co', path='Ieeg_signal/CO', num=0):
        # plt.figure(figsize=(100,5))
        # ax = plt.axes(frameon=True)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        if num == 0:
            path += '.npy'
        else:
            path += '_clip_{}.npy'.format(num)
        fs = np.load(path)

        ids = []
        for i in range(len(fs)-1):
            if fs[i] != 0 and fs[i+1] == 0:
                #print(i)
                ids.append(i)
            if fs[i] == 0 and fs[i+1] != 0:
                #print(i+1)
                ids.append(i+1)

        # print(ids)

        del_list = []
        single_missing = []
        gap_missing = []
        for start_prior, end_after in zip(ids[0::2], ids[1::2]):
            anchor_start = fs[start_prior]
            anchor_end = fs[end_after]
            # print('gap start idx:{}, value:{}'.format(start_prior, anchor_start))
            # print('gap end idx:{}, value:{}'.format(end_after, anchor_end))
            range_start = 0
            range_end = 0

            for idx in range(start_prior, start_prior-500, -1):
                if abs(fs[idx] - anchor_end) < 0.01:
                    range_start = start_prior - idx
                    # print('backward: idx:{}, range:{}, value:{}'.format( idx, range_start, fs[idx] ))
                    break
                                                
            for idx in range(end_after, end_after+500):
                if abs(fs[idx] - anchor_start) < 0.01:
                    range_end = idx - end_after
                    # print('forward: idx:{}, range:{}, value:{}'.format( idx, range_end, fs[idx] ))
                    break
            
            if range_start != 0 or range_end != 0:
                print(range_start, range_end)
                if range_start == 0:
                    range_start = 500
                if range_end == 0:
                    range_end = 500
                # print('start:{}, back:{}'.format(start_prior, start_prior-range_start))
                # print('end:{}, forward:{}'.format(end_after, end_after+range_end))
                if range_start > range_end:
                    del_list.append((start_prior+1,end_after+range_end+1))
                elif range_start < range_end:
                    del_list.append((start_prior-range_start,end_after))
                elif range_start == range_end:
                    del_list.append((start_prior+1,end_after+range_end+1))
                # print('after clipping: start_idx_{}, end_idx_{}'.format(del_list[-1][0], del_list[-1][1]))
            else:
                #print('NEED check! start after:{}, end before:{}'.format(start_prior, end_after))
                if end_after - start_prior <= 2:     
                    single_missing.append(start_prior+1)   
                else:
                    gap_missing.append((start_prior,end_after))
                    
            # print('\t')     
        print('gap delete', del_list)
        print('point delete', single_missing)
        print('gap remove', gap_missing)

        # for pair in reversed(del_list):
        #     for idx in range(pair[1]-1, pair[0]-1, -1):
        #         fs = np.delete(fs, idx)

        # for idx in reversed(single_missing):
        #     fs = np.delete(fs, idx)

        # for pair in reversed(gap_missing):
        #     for idx in range(pair[1]-1, pair[0], -1):
        #         fs = np.delete(fs, idx)
        


        # if sig_type == 'o3':
        #     np.save('Ieeg_signal/O3_clip_{}.npy'.format(num+1), fs)
        # elif sig_type == 'no2':
        #     np.save('Ieeg_signal/NO2_clip_{}.npy'.format(num+1), fs)
        # elif sig_type == 'co':
        #     np.save('Ieeg_signal/CO_clip_{}.npy'.format(num+1), fs)

        # plt.figure(figsize=(100,5))
        # ax = plt.axes(frameon=True)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # length = int(len(fs))
        # xs = np.linspace(0,length,length) 
        # if sig_type == 'o3':
        #     plot_title = 'O3_clipped_{}'.format(num+1)
        # elif sig_type == 'no2':
        #     plot_title = 'NO2_clipped_{}'.format(num+1)
        # elif sig_type == 'co':
        #     plot_title = 'CO_clipped_{}'.format(num+1)
        # plt.plot(xs, fs, 'gray')
        # plt.title(plot_title)
        # plt.savefig('jul9/{}.jpg'.format(plot_title))
        # print('single plot {} saved'.format(plot_title))
        # plt.close()



    # tailor(num=3)

    path = 'Ieeg_signal/NO2_clip_4.npy'
    fs = np.load(path)
    print(len(fs))
    num = len(fs) // 512
    for i in range(num):
        out = fs[i*512:(i+1)*512]
        print('start {}, end {}'.format(i*512, (i+1)*512))
        np.save('Ieeg_signal/NO2_clip_512_{}.npy'.format(i+1), out)