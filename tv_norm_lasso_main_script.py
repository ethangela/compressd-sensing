import math
import numpy as np
import sys
from argparse import ArgumentParser
import os
import random
from scipy.fftpack import fft, ifft
import pandas as pd
from skimage.io import imread, imsave

# tv norm
from pyunlocbox import functions
from pyunlocbox import solvers

# lasso
from sklearn.linear_model import Lasso
from l1regls import l1regls
from cvxopt import matrix
import pywt

# == functions == #
def random_flip(value):
    a = []
    for _ in range(value):
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
    return img

def load_2D(path, img_name):
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    return img
    
def load_img(path, img_name, decoder_type):
    img_path = os.path.join(path, img_name)
    img = imread(img_path)
    img = np.transpose(img, (1, 0, 2))
    img = img[25:153,45:173,:] 
    img_clean = img / 255.
    return img_clean
     
def get_l2_loss(image1, image2, image_mode, decoder_type):
    assert image1.shape == image2.shape
    # if image_mode == '3D' and decoder_type == 'lasso_wavelet': !!!!!!!!!!!
    #     image1, image2 = image1[0,23:201,3:221], image2[0,23:201,3:221]
    return np.mean((image1 - image2)**2)
    
def load_mask(path, img_name, image_mode, decoder_type): 
    img_path = os.path.join(path, img_name)
    img = np.load(img_path)
    if len(img.shape) == 1:
        img = img[:, None]
    if image_mode == '3D':
        # if decoder_type == 'lasso_wavelet':
        #     img = np.pad(img, ((23,23), (3,3)), 'constant', constant_values=0) #[178,218] -> [224,224] !!!!!!!!!!!
        img = img[:, :, None]
    return img

def create_A_selection(signal_size, measurement_size):
    mask_ = np.zeros((1,signal_size))
    idx_list = np.random.choice(signal_size, measurement_size, replace=False)
    for idx in idx_list:
        mask_[:,idx] = 1.0 
    return mask_

def save_out_img(img, path, img_name, decoder_type, model_type, num_mea_info, mask_info, noise_level, image_mode): 
    if image_mode == '1D':
        img = img[:,0]
        if model_type == 'inpainting':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_mask_' + mask_info +'.npy'
        elif model_type == 'denoising':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_noise_' + noise_level +'.npy'
        elif model_type == 'compressing':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_num_mea_' + num_mea_info +'.npy'
        np.save(os.path.join(path, name), img)
    elif image_mode == '2D':
        if model_type == 'inpainting':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_mask_' + mask_info +'.npy'
        elif model_type == 'denoising':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_noise_' + noise_level +'.npy'
        elif model_type == 'compressing':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_num_mea_' + num_mea_info +'.npy'
        np.save(os.path.join(path, name), img)
    elif image_mode == '3D':
        # if decoder_type == 'lasso_wavelet':
        #     img = img[0,23:201,3:221] !!!!!!!!!!!
        img = img * 255.
        img = np.transpose(img, (1, 0, 2))
        if model_type == 'inpainting':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_mask_' + mask_info +'.jpg'
        elif model_type == 'denoising':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_noise_' + noise_level +'.jpg'
        elif model_type == 'compressing':
            name = img_name[:-4] + '_decoder_type_' + decoder_type + '_model_type_' + model_type + '_num_mea_' + num_mea_info +'.jpg'
        imsave(os.path.join(path, name), img.astype(np.uint8))
        print('saving complete')
    print('saving shape {}'.format(img.shape))



# == main operation == #
def main_tv(hparams):

    ## === Set up=== ##
    # Printer setup
    #sys.stdout = open(hparams.text_file_path, 'w')

    # Get inputs
    if hparams.image_mode == '1D':
        x_real = np.array(load_1D(hparams.path, hparams.img_name)).astype(np.float32) #[4096,1]
    elif hparams.image_mode == '2D':
        x_real = np.array(load_2D(hparams.path, hparams.img_name)).astype(np.float32) #[64,64]
    elif hparams.image_mode == '3D':
        x_real = np.array(load_img(hparams.path, hparams.img_name, hparams.decoder_type)).astype(np.float32) #[178,218,3] /  [224,224,3]
    
    # Initialization
    #np.random.seed(7)
    sig_shape = x_real.shape[0]*x_real.shape[1] #n = 4096*1 or 64*64 or 178*218 or 224*224
    random_vector = None #initialization 
    A = None #initialization
    selection_mask = None #initialization
    random_arr = random_flip(sig_shape) #initialization #[n,]
    mask = None #initialization
    
    # Get measurement matirx
    if hparams.model_type == 'denoising' or hparams.model_type == 'compressing':
        if hparams.type_measurements == 'random': #compressed sensing
            if hparams.image_mode != '3D':
                A = np.random.randn(hparams.num_measurements, sig_shape).astype(np.float32)#[m,n]
                noise_shape = [hparams.num_measurements,1]#[m,1]
            else:
                A = np.random.randn(int(hparams.num_measurements/3), sig_shape).astype(np.float32)#[m,n]
                noise_shape = [int(hparams.num_measurements/3),1]#[m,1]
        elif hparams.type_measurements == 'identity': #denoising 
            A = np.identity(sig_shape).astype(np.float32)#[n,n]
            noise_shape = [sig_shape,1]#[n,1]
            observ_noise = hparams.noise_level * np.random.randn(noise_shape[0], noise_shape[1])#[n,1]
        elif hparams.type_measurements == 'circulant': #compressed sensing
            if hparams.image_mode != '3D':
                random_vector = np.random.normal(size=sig_shape)#[n,]  
                selection_mask = create_A_selection(sig_shape, hparams.num_measurements)#[1,n]  
            else:
                random_vector = np.random.normal(size=sig_shape)#[n,]  
                selection_mask = create_A_selection(sig_shape, int(hparams.num_measurements/3))#[1,n]  
        
            def circulant_np(signal_vector, random_arr_p = random_arr.reshape(-1,1), random_vector_p = random_vector.reshape(-1,1), selection_mask_p = selection_mask.reshape(-1,1)): 
                #step 0: Flip
                signal_vector = signal_vector * random_arr_p #[n,1] * [n,1] -> [n,1]
                #step 1: F^{-1} @ x
                r1 = ifft(signal_vector)#[n,1]
                #step 2: Diag() @ F^{-1} @ x
                Ft = fft(random_vector_p)#[n,1]
                r2 = np.multiply(r1, Ft)#[n,1] * [n,1] -> [n,1]
                #step 3: F @ Diag() @ F^{-1} @ x
                compressive = fft(r2)#[n,1]
                #step 4:  R_{omega} @ C_{t} @ D){epsilon}
                compressive = compressive.real#[n,1] 
                select_compressive = compressive * selection_mask_p#[n,1] * [n,1] -> [n,1]
                return select_compressive
    
    elif hparams.model_type == 'inpainting':
        if hparams.image_mode == '1D':
            mask = load_mask('Masks', hparams.mask_name_1D, hparams.image_mode, hparams.decoder_type)#[n,1]
        elif hparams.image_mode == '2D' or hparams.image_mode == '3D':
            mask = load_mask('Masks', hparams.mask_name_2D, hparams.image_mode, hparams.decoder_type)#[n,n]

    
    ## === TV norm === ##
    if hparams.decoder_type == 'tv_norm':
        # Construct observation and perform reconstruction
        if hparams.model_type == 'inpainting':
            # measurements and observation
            g = lambda x: mask * x #[4096,1] * [4096,1] / [178,218,3] * [178,218,3]
            y_real = g(x_real) #[4096,1] / [178,218,3]
            # tv norm
            if hparams.image_mode == '1D':
                f1 = functions.norm_tv(dim=1)
            elif hparams.image_mode == '2D':
                f1 = functions.norm_tv(dim=2)
            elif hparams.image_mode == '3D':
                f1 = functions.norm_tv(dim=3)
            # L2 norm
            tau = hparams.tau
            f2 = functions.norm_l2(y=y_real, A=g, lambda_=tau)
            # optimisation
            solver = solvers.forward_backward(step=0.5/tau)
            x0 = np.array(y_real)  # Make a copy to preserve im_masked.
            ret = solvers.solve([f1, f2], x0, solver, maxit=3000) #output = ret['sol']
            # output
            out_img = ret['sol'] #[4096,1] / [178,218,3]
        elif hparams.model_type == 'denoising':
            assert hparams.type_measurements == 'identity'
            if hparams.image_mode == '3D':
                out_img_list = []
                for i in range(x_real.shape[-1]):
                    # measurements and observation
                    y_real = np.matmul(A, x_real[:,:,i].reshape(-1,1)) + observ_noise # [n,n] * [n,1] -> [n,1]
                    # tv norm
                    f1 = functions.norm_tv(dim=1)
                    # epsilon
                    N = math.sqrt(sig_shape)
                    epsilon = N * hparams.noise_level
                    # L2 ball 
                    y = np.reshape(y_real, -1) #[n,1] -> [n,]
                    f = functions.proj_b2(y=y, epsilon=epsilon) 
                    f2 = functions.func()
                    # Indicator functions
                    f2._eval = lambda x: 0
                    def prox(x, step):
                        return np.reshape(f.prox(np.reshape(x, -1), 0), y_real.shape)
                    f2._prox = prox
                    # solver
                    solver = solvers.douglas_rachford(step=0.1)
                    x0 = np.array(y_real)#[n,1]
                    ret = solvers.solve([f1, f2], x0, solver) 
                    # output
                    out_img_piece = ret['sol'].reshape(x_real.shape[0], x_real.shape[1]) #[178,218]
                    out_img_list.append(out_img_piece)
                out_img = np.transpose(np.array(out_img_list), (1, 2, 0))
            else:
                # measurements and observation
                y_real = np.matmul(A, x_real.reshape(-1,1)) + observ_noise # [n,n] * [n,1] -> [n,1]
                # tv norm
                f1 = functions.norm_tv(dim=1)
                # epsilon
                N = math.sqrt(sig_shape)
                epsilon = N * hparams.noise_level
                # L2 ball 
                y = np.reshape(y_real, -1) #[n,1] -> [n,]
                f = functions.proj_b2(y=y, epsilon=epsilon) 
                f2 = functions.func()
                # Indicator functions
                f2._eval = lambda x: 0
                def prox(x, step):
                    return np.reshape(f.prox(np.reshape(x, -1), 0), y_real.shape)
                f2._prox = prox
                # solver
                solver = solvers.douglas_rachford(step=0.1)
                x0 = np.array(y_real)#[n,1]
                ret = solvers.solve([f1, f2], x0, solver) 
                # output
                out_img = ret['sol']#[n,1]
        elif hparams.model_type == 'compressing': 
            assert hparams.type_measurements == 'circulant'
            if hparams.image_mode == '3D':
                out_img_list = []
                for i in range(x_real.shape[-1]):
                    # construct observation
                    g = circulant_np
                    y_real = g(x_real[:,:,i].reshape(-1,1)) #[n,1] -> [n,1]
                    # tv norm
                    f1 = functions.norm_tv(dim=1)
                    # L2 norm
                    tau = hparams.tau
                    f2 = functions.norm_l2(y=y_real, A=g, lambda_=tau)
                    # optimisation solver
                    A_real = np.random.normal(size=(int(hparams.num_measurements/3), sig_shape))
                    step = 0.5 / np.linalg.norm(A_real, ord=2)**2
                    solver = solvers.forward_backward(step=step) #solver = solvers.forward_backward(step=0.5/tau)
                    # initialisation
                    x0 = np.array(y_real) #[n,1]
                    # output
                    ret = solvers.solve([f1, f2], x0, solver, rtol=1e-4, maxit=3000) #output = ret['sol']
                    out_img_piece = ret['sol'].reshape(x_real.shape[0], x_real.shape[1]) #[178,218]
                    out_img_list.append(out_img_piece)
                out_img = np.transpose(np.array(out_img_list), (1, 2, 0))
            else:
                # construct observation
                g = circulant_np
                y_real = g(x_real.reshape(-1,1)) #[n,1] -> [n,1]
                # tv norm
                f1 = functions.norm_tv(dim=1)
                # L2 norm
                tau = hparams.tau
                f2 = functions.norm_l2(y=y_real, A=g, lambda_=tau)
                # optimisation solver
                A_real = np.random.normal(size=(hparams.num_measurements, sig_shape))
                step = 0.5 / np.linalg.norm(A_real, ord=2)**2
                solver = solvers.forward_backward(step=step) #solver = solvers.forward_backward(step=0.5/tau)
                # initialisation
                x0 = np.array(y_real) #[n,1]
                # output
                ret = solvers.solve([f1, f2], x0, solver, rtol=1e-4, maxit=3000) #output = ret['sol']
                out_img = ret['sol']#[n,1]

    

    # ## === Lasso  wavelet === ##
    elif hparams.decoder_type == 'lasso_wavelet':
        # Construct lasso wavelet functions
        def solve_lasso(A_val, y_val, hparams): #(n,m), (1,m)
            if hparams.lasso_solver == 'sklearn': 
                lasso_est = Lasso(alpha=hparams.lmbd)
                lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
                x_hat = lasso_est.coef_
                x_hat = np.reshape(x_hat, [-1])
            elif hparams.lasso_solver == 'cvxopt':
                A_mat = matrix(A_val.T) #[m,n]
                y_mat = matrix(y_val.T) ###
                x_hat_mat = l1regls(A_mat, y_mat) 
                x_hat = np.asarray(x_hat_mat) 
                x_hat = np.reshape(x_hat, [-1]) #[n, ]
            elif hparams.lasso_solver == 'pyunlocbox':
                tau = hparams.tau
                f1 = functions.norm_l1(lambda_=tau) 
                f2 = functions.norm_l2(y=y_val.T, A=A_val.T)
                if hparams.model_type == 'compressing':
                    if hparams.image_mode == '3D':
                        A_real = np.random.normal(size=(int(hparams.num_measurements/3), sig_shape))
                    else:
                        A_real = np.random.normal(size=(hparams.num_measurements, sig_shape))
                    step = 0.5 / np.linalg.norm(A_real, ord=2)**2
                else:
                    step = 0.5 / np.linalg.norm(A_val, ord=2)**2 
                solver = solvers.forward_backward(step=step)
                x0 = np.zeros((sig_shape,1))
                ret = solvers.solve([f1, f2], x0, solver, rtol=1e-4, maxit=3000)
                x_hat_mat = ret['sol']
                x_hat = np.asarray(x_hat_mat) 
                x_hat = np.reshape(x_hat, [-1]) #[n, ]
            return x_hat

        #generate basis
        def generate_basis(size):
            """generate the basis"""
            x = np.zeros((size, size)) 
            coefs = pywt.wavedec2(x, 'db1')
            n_levels = len(coefs)
            basis = []
            for i in range(n_levels):
                coefs[i] = list(coefs[i])
                n_filters = len(coefs[i])
                for j in range(n_filters):
                    for m in range(coefs[i][j].shape[0]):
                        try:
                            for n in range(coefs[i][j].shape[1]):
                                coefs[i][j][m][n] = 1
                                temp_basis = pywt.waverec2(coefs, 'db1')
                                basis.append(temp_basis)
                                coefs[i][j][m][n] = 0
                        except IndexError:
                            coefs[i][j][m] = 1
                            temp_basis = pywt.waverec2(coefs, 'db1')
                            basis.append(temp_basis)
                            coefs[i][j][m] = 0
            basis = np.array(basis)
            return basis
                
        
        def wavelet_basis(path_): 
            if path_ == 'Ieeg_signal':
                W_ = generate_basis(32)
                W_ = W_.reshape((1024, 1024))
            elif path_ == 'Celeb_signal':
                W_ = generate_basis(128)
                W_ = W_.reshape((16384, 16384))
            else:    
                W_ = generate_basis(64)
                W_ = W_.reshape((4096, 4096)) 
            return W_

        def lasso_wavelet_estimator(A_val, y_val, hparams): #(n,m), (1,m)
            W = wavelet_basis(hparams.path) #[n,n] 
            if not callable(A_val):
                WA = np.dot(W, A_val) #[n,n] * [n,m] = [n,m] 
            else:
                WA = np.array([A_val(W[i,:].reshape(-1,1)).reshape(-1) for i in range(len(W))]) #[n,n] -> [n,n] 
            z_hat = solve_lasso(WA, y_val, hparams) # [n, ]
            x_hat = np.dot(z_hat, W) #[n, ] * [n,n] = [n, ]
            x_hat_max = np.abs(x_hat).max() 
            x_hat = x_hat / (1.0 * x_hat_max)
            return x_hat

        # Construct inpainting masks 
        def get_A_inpaint(mask_p): 
            mask = mask_p.reshape(1, -1)
            A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
            A = np.asarray([a for a in A if np.sum(a) != 0])
            A = np.sqrt(sig_shape) * A # Make sure that the norm of each row of A is sig_shape
            assert all(np.abs(np.sum(A**2, 1) - sig_shape) < 1e-6)
            return A.T    

        # Perofrm reconstruction
        if hparams.model_type == 'inpainting':
            # measurements and observation
            A_val = get_A_inpaint(mask) #(n,m)
            if hparams.image_mode == '3D':
                out_img_list = []
                for i in range(x_real.shape[-1]):
                    y_real = np.matmul(x_real[:,:,i].reshape(1,-1), A_val) #(1,m)
                    out_img_piece = lasso_wavelet_estimator(A_val, y_real, hparams)
                    out_img_piece = out_img_piece.reshape(x_real.shape[0], x_real.shape[1])
                    out_img_list.append(out_img_piece)
                out_img = np.transpose(np.array(out_img_list), (1, 2, 0))
            elif hparams.image_mode == '1D':
                y_real = np.matmul(x_real.reshape(1,-1), A_val) #(1,m)
                out_img = lasso_wavelet_estimator(A_val, y_real, hparams)
                out_img = out_img.reshape(-1,1)
        elif hparams.model_type == 'denoising':
            assert hparams.type_measurements == 'identity'
            A_val = A #(n,n)
            if hparams.image_mode == '3D':
                out_img_list = []
                for i in range(x_real.shape[-1]):
                    y_real = x_real[:,:,i].reshape(1,-1) + observ_noise.T  
                    out_img_piece = lasso_wavelet_estimator(A_val, y_real, hparams)
                    out_img_piece = out_img_piece.reshape(x_real.shape[0], x_real.shape[1])
                    out_img_list.append(out_img_piece)
                out_img = np.transpose(np.array(out_img_list), (1, 2, 0))
            elif hparams.image_mode == '1D':
                y_real = np.matmul(x_real.reshape(1,-1), A_val) + observ_noise.T 
                out_img = lasso_wavelet_estimator(A_val, y_real, hparams)
                out_img = out_img.reshape(-1,1)
        elif hparams.model_type == 'compressing': 
            assert hparams.type_measurements == 'circulant'
            A_val = circulant_np
            if hparams.image_mode == '3D':
                out_img_list = []
                for i in range(x_real.shape[-1]):
                    y_real = A_val(x_real[:,:,i].reshape(-1,1)).reshape(1,-1)#[n,1] -> [1,n]
                    out_img_piece = lasso_wavelet_estimator(A_val, y_real, hparams)
                    out_img_piece = out_img_piece.reshape(x_real.shape[0], x_real.shape[1])
                    out_img_list.append(out_img_piece)
                out_img = np.transpose(np.array(out_img_list), (1, 2, 0))
            elif hparams.image_mode == '1D':
                y_real = A_val(x_real).reshape(1,-1)#[n,1] -> [1,n]
                out_img = lasso_wavelet_estimator(A_val, y_real, hparams)
                out_img = out_img.reshape(-1,1)


    ## === Printer === ##
    # Compute and print measurement and l2 loss
    # if hparams.image_mode == '3D' and hparams.model_type != 'inpainting':
    #     x_real = x_real.reshape(-1,1)
    l2_losses = get_l2_loss(out_img, x_real, hparams.image_mode, hparams.decoder_type)
    psnr = 10 * np.log10(1 * 1 / l2_losses) #PSNR

    # Printer info
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
        num_mea_info = str(hparams.num_measurements)
        noise_level_info = 'NA'
    elif hparams.model_type == 'denoising':
        mask_info = 'NA'
        type_mea_info = 'NA'
        num_mea_info = 'NA'
        noise_level_info = str(hparams.noise_level)

    # Print result 
    print ('Final representation PSNR for img_name:{}, model_type:{}, type_mea:{}, num_mea:{}, mask:{}, decoder:{} tau:{} noise:{} is {}'.format(hparams.img_name, 
            hparams.model_type, type_mea_info, num_mea_info, mask_info, 
            hparams.decoder_type, hparams.tau, noise_level_info, psnr))
    print('END')
    print('\t')
    #sys.stdout.close()


    ## == to pd frame == ## 
    if hparams.pickle == 1:
        pickle_file_path = hparams.pickle_file_path
        if not os.path.exists(pickle_file_path):
            d = {'img_name':[hparams.img_name], 'model_type':[hparams.model_type], 'type_mea':[type_mea_info], 'num_mea':[num_mea_info], 'mask_info':[mask_info], 
                'decoder_type':[hparams.decoder_type], 'tau':[hparams.tau], 'noise':[noise_level_info], 'psnr':[psnr]}
            df = pd.DataFrame(data=d)
            df.to_pickle(pickle_file_path)
        else:
            d = {'img_name':hparams.img_name, 'model_type':hparams.model_type, 'type_mea':type_mea_info, 'num_mea':num_mea_info, 'mask_info':mask_info, 
                'decoder_type':hparams.decoder_type, 'tau':hparams.tau, 'noise':noise_level_info, 'psnr':psnr}
            df = pd.read_pickle(pickle_file_path)
            df = df.append(d, ignore_index=True)
            df.to_pickle(pickle_file_path)


    ## === Save === ##
    if hparams.save == 1:
        save_out_img(out_img, 'result/', hparams.img_name, hparams.decoder_type, hparams.model_type, num_mea_info, mask_info, noise_level_info, hparams.image_mode)



if __name__ == '__main__':
    PARSER = ArgumentParser()
 
    # Input
    PARSER.add_argument('--image_mode', type=str, default='1D', help='path stroing the images') 
    PARSER.add_argument('--path', type=str, default='Gaussian_signal', help='path stroing the images')
    PARSER.add_argument('--noise_level', type=float, default=0.05, help='std dev of noise') 
    PARSER.add_argument('--img_name', type=str, default='1D_exp_0.25_4096_20.npy', help='image to use') 
    PARSER.add_argument('--model_type', type=str, default='', help='inverse problem model type')
    PARSER.add_argument('--type_measurements', type=str, default='circulant', help='measurement type') 
    PARSER.add_argument('--num_measurements', type=int, default=500, help='number of gaussian measurements') 
    PARSER.add_argument('--mask_name_1D', type=str, default='1D_mask_block_4096_8_1.npy', help='mask to use') 
    PARSER.add_argument('--mask_name_2D', type=str, default='', help='mask to use') 
    PARSER.add_argument('--pickle_file_path', type=str, default='tv_normfeb40.pkl') 
    PARSER.add_argument('--pickle', type=int, default=1) 
    PARSER.add_argument('--save', type=int, default=0) 

    # Decoder 
    PARSER.add_argument('--decoder_type', type=str, default='tv_norm', help='decoder type') 
    PARSER.add_argument('--lasso_solver', type=str, default='pyunlocbox', help='lasso solver type') 

    
    # "Training"
    PARSER.add_argument('--tau', type=float, default=100, help='inpainting trade-off') 
   
    HPARAMS = PARSER.parse_args()
    
    main_tv(HPARAMS)



