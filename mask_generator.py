import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import random

def mask(size, mtype='block', prob_str='2', blksize=16, dim=1, num=1):
    if len(prob_str) > 1:
        prob_list = prob_str.split('/')
        print(prob_list)
        prob = float(int(prob_list[1])/int(prob_list[0]))
        prob_printer = prob_list[1] + '_' + prob_list[0]
    else:
        prob = float(1/int(prob_str))
        prob_printer = prob_str

    if dim == 1:
        m = np.ones(size)
        count = 0
        if mtype == 'block':
            num_chunk = int(size/blksize)
            for i in range(num_chunk):
                if random.randint(1, 10000) <= int(10001*prob):
                    m[16*i:16*(i+1)] = 0
                    count+=1
            print('{} out of {} are block-masked'.format(count, num_chunk))
        elif mtype == 'random':
            for i in range(size):
                if random.randint(1, 10000) <= int(10001*prob):
                    m[i] = 0
                    count+=1
                else:
                    continue
            print('{} out of {} are random-masked'.format(count, size))
        np.save('Masks/1D_mask_'+mtype+'_'+str(size)+'_'+prob_printer+'_'+str(num)+'.npy', m)
    elif dim == 2:
        m = np.ones(size)
        count = 0
        if mtype == 'block':
            blck_width = int(size[0]/blksize) #11
            blck_height = int(size[1]/blksize) #13
            #t_w = int((size[0] - (blck_width * blksize))/2) #1
            #t_h = int((size[1] - (blck_height * blksize))/2) #5
            for i in range(blck_width):
                for j in range(blck_height):
                    if random.randint(1, 10000) <= int(10001*prob):
                        m[blksize*i:blksize*(i+1), blksize*j:blksize*(j+1)] = 0 #m[t_w+16*i:t_w+16*(i+1), t_h+16*j:t_h+16*(j+1)] = 0 
                        count+=1
            print('{} out of {} are block-masked'.format(count, blck_width*blck_height))
        elif mtype == 'random':
            for i in range(size[0]):
                for j in range(size[1]):
                    if random.randint(1, 10000) <= int(10001*prob):
                        m[i,j] = 0
                        count+=1
            print('{} out of {} are random-masked'.format(count, size[0]*size[1]))
        np.save('Masks/2D_mask_'+mtype+'_'+str(size[0])+'_'+str(size[1])+'_'+prob_printer+'_'+str(num)+'.npy', m)



if __name__ == '__main__':
    for type_ in ['random', 'block']:
        for prob_ in ['10/9', '4/3', '2', '4', '8']:
            for num_ in range(1,31):
                #mask(1024, mtype=type_, prob_str=prob_, blksize=16, dim=1, num=num_)
                mask((128,128), mtype=type_, prob_str=prob_, blksize=8, dim=2, num=num_)
    