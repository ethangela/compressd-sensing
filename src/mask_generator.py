import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
import random

def mask(size, mtype='block', prob=2, blksize=16, num=1):
    m = np.ones(size)
    count = 0
    if mtype == 'block':
        num_chunk = int(size/blksize)
        num_block = int(num_chunk/prob)
        for i in range(num_chunk):
            if random.randint(1, 1000) <= int(1001/prob):
                m[16*i:16*(i+1)] = 0
                count+=1
        print('{} out of {} are block-masked'.format(count, num_block))
    #elif mtype == 'random':

    np.save('../mask/1D_mask_'+mtype+'_'+str(size)+'_'+str(prob)+'_'+str(num)+'.npy', m)

if __name__ == '__main__':
    mask(4096, prob=8)