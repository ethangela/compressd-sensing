# compressd-sensing

This repository provides codes for project of Compressed Sensing with Deep Decoder Prior in implementation of both tensorflow and pytorch (Torch implementation is depreciated).

## Requirements
Tensorflow 1.13+ and Python 3.5+. Other dependecies include numpy, matplotlib, pandas and scipy etc.

## Usage
# For the purpose of inapinting expriment:

python src/compressed_sensing_for_image_signal.py --model_type inpainting --mask_name_1D 1D_mask_block_4096_2_1.npy --img_name 1D_exp_0.25_1.npy --k 300 --num_layers 22 --input_size 50 --filter_size 8 --pickle_file_path nov30_block_ipt_exp.pkl

# For the purpose of denoising (with circulant technique applied) expriment:

python src/compressed_sensing_for_image_signal.py --model_type denoising --type_measurements circulant --num_measurements 500 --img_name 1D_exp_0.25_1.npy --k 300 --num_layers 22 --input_size 50 --filter_size 8 --pickle_file_path nov30_block_ipt_exp.pkl


