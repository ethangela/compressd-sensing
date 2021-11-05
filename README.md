# On Architecture Selection for Linear InverseProblems with Untrained Neural Networks

This repository provides codes for paper "On Architecture Selection for Linear InverseProblems with Untrained Neural Networks". The codes are built for the purpose of solving linear inverse problems using various priors, e.g. Deep Decoder, Deep Image Prior, and TV Normlization, and optimizing the hyperparameters of these priors. 

Some parts of the codes in tv_norm_lasso_main_script.py are taken from https://github.com/AshishBora/csgm.

## Requirements
Tensorflow 1.13+ and Python 3.5+. Other dependecies include numpy, matplotlib, pandas and scipy etc.

## Example Script Command For Usage: 
#### I. For the purpose of hyperparameter selection training:

##### 1) synthetic signal inpainting: 
python hyperparameter_selection_main_script.py --model_type inpainting --mask_name_1D 1D_mask_block_1024_2_1.npy --image_mode 1D --path Gaussian_signal --img_name 1D_rbf_1.0_4096_30.npy

##### 2) real air data denoising: 
python hyperparameter_selection_main_script.py --model_type denoising --type_measurements identity --noise_level 0.05 --image_mode 1D --path Air_signal --img_name co_1024_1.npy

##### 3) celebA data compressing: 
python hyperparameter_selection_main_script.py --model_type compressing --num_measurements 5000 --image_mode 2D --path Celeb_signal --img_name 182656.jpg

##### 4) MRI data compressing: 
python MRI_hyperparameter_selection_main_script.py.py --arch_name ConvDecoder --stype successive_halving --file_name file1000758.h5

#### II. For the purpose of single sample testing:

##### 1) synthetic signal / real air signal / celeba signal: 
python linear_inverse_problem_main_script.py --model_type inpainting --mask_name_1D 1D_mask_block_4096_2_1.npy --chn 200 --lay 3 --ipt 10 --lrn 0.0015 --img_name 1D_exp_0.25_4096_30.npy

##### 2) MRI signal only: 
python linear_inverse_problem_MRI_main_script.py --arch_name DD --chn 368 --lay 10 --ipt 16 --lrn 0.008 --file_name file1000308.h5


#### III. For the purpose of Gaussian signal generation:

python Gaussian_signal_generator.py


#### IV. For the purpose of mask generation:

python mask_generator.py


