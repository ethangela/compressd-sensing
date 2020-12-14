import numpy as np 
from scipy.stats import multivariate_normal
from argparse import ArgumentParser
from scipy.special import gamma, kv

def matern_kernel(r, l = 1, v = 1):
    r[r == 0] = 1e-8
    part1 = 2 ** (1 - v) / gamma(v)
    part2 = (np.sqrt(2 * v) * r / l) ** v
    part3 = kv(v, np.sqrt(2 * v) * r / l)
    return part1 * part2 * part3

def sample_from_1D_Gaussian(size, kernel, l_val, idx):
    
    #sample points range 
    m = size  
    xs = np.linspace(-10,10,m) #Range vector (101,)
    
    #mean vetor 
    mxs = np.zeros(m) #Zero mean vector (101,)

    #covariance by kernel
    l = l_val #hyperparameters, i.e. 2, sqrt(0.5), 1
    sigma_square = 1 #hyperparameters
    if kernel == 'rbf':
        Kss = sigma_square * np.exp( -1/(l**2) * np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T)**2 ) #Covariance matrix
    elif kernel == 'exp':
        Kss = np.exp( -1/l * np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T) ) #Covariance matrix
    elif kernel == 'peri':
        Kss = np.exp( -2/(l**2) * np.sin(np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T)/2)**2 ) #Covariance matrix
    elif kernel == 'matern':
        Kss = matern_kernel(r=np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T), l=l)
    print('step 0 complete')
    
    #sample 
    s = 10 #number of drawing samples from the prior
    fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular=True).rvs(s).T
    print('step 1 complete')
    print(fs.shape)
    r = np.random.randint(s, size=1)
    fs = fs[:, r[0]]
    print('step 2 complete')
    
    #scale to range(0,1)
    xs = (xs - np.min(xs))/np.ptp(xs)
    fs = (fs - np.min(fs))/np.ptp(fs)
    print(fs.shape)
    
    #save 
    ##plt.plot(xs, fs, 'gray') # Plot the samples
    ##plt.savefig('1D_'+kernel+'_'+str(l_val)+'.jpg')
    #np.save('1D_'+kernel+'_'+str(l_val)+'.npy', fs)
    np.save('../image/Gaussian signal/1D_'+kernel+'_'+str(l_val)+'_'+str(size)+'_'+str(idx)+'.npy', fs)
   
   
def main(size, hparams, idx):
    return sample_from_1D_Gaussian(size, hparams.kernel, hparams.length, idx) 


if __name__ == '__main__':
    
    size_list = [int(32*pow(p, l)) for p in [1.5, 2, 3, 4] for l in [2,3,4,5]][:-1]
    
    for num in range(1,6):
        for i in range(len(size_list)):
            for kernel in [('exp',0.25), ('matern',1.0), ('rbf',3.0)]:
                sample_from_1D_Gaussian(size_list[i], kernel[0], kernel[1], num)
    
