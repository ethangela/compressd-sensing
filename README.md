# On Architecture Selection for Linear InverseProblems with Untrained Neural Networks

This repository provides codes for paper "On Architecture Selection for Linear InverseProblems with Untrained Neural Networks". The codes are built for the purpose of solving linear inverse problems using various priors, e.g. Deep Decoder, Deep Image Prior, and TV Normlization, and optimizing the hyperparameters of these priors. 

Some parts of the codes in tv_norm_lasso_main_script.py are taken from https://github.com/AshishBora/csgm.

## Requirements
Tensorflow 1.13+ and Python 3.5+. Other dependecies include numpy, matplotlib, pandas and scipy etc.

## Usage
#### For the purpose of liner inverse problem expriment (e.g., inpainting):

python linear_inverse_problem_main_script.py --model_type inpainting --mask_name_1D 1D_mask_block_4096_2_1.npy --img_name 1D_exp_0.25_4096_30.npy

#### For the purpose of hyperparameter selection (e.g., denoising):

python hyperparameter_selection_main_script.py --model_type denoising --type_measurements identity --noise_level 0.05 --img_name 1D_rbf_1.0_4096_30.npy

#### For the purpose of Gaussian signal generation:

python Gaussian_signal_generator.py

#### For the purpose of mask generation:

python mask_generator.py


