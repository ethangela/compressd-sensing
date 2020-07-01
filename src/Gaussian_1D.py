import numpy as np 
import numpy.matlib
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy.stats import truncnorm
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
import numpy as np
import matplotlib .pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


def sample_from_1D_Gaussian(size, kernel, l_val):
    
    #sample points range 
    m = size*size
    xs = np. linspace (-10,10,m) #Range vector (101,)
    
    #mean vetor 
    mxs = np.zeros(m) #Zero mean vector (101,)

    #rbf kernel
    l = l_val #hyperparameters #2, sqrt(0.5), 1
    sigma_square = 1 #hyperparameters
    if kernel == 'rbf':
        Kss = sigma_square * np.exp( -1/(l**2) * np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T)**2 ) #Covariance matrix
    elif kernel == 'exp':
        Kss = np.exp( -1/l * np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T) ) #Covariance matrix
    elif kernel == 'peri':
        Kss = np.exp( -2/(l**2) * np.sin(np.abs(xs[:,np.newaxis]-xs[:,np.newaxis].T)/2)**2 ) #Covariance matrix
    print('done 0')
    
    #sample 
    s = 10 #number of drawing samples from the prior
    fs = multivariate_normal(mean=mxs ,cov=Kss , allow_singular =True).rvs(s).T
    print('done 1')
    print(fs.shape)
    r = np.random.randint(s, size=1)
    fs = fs[:, r[0]]
    print('done 2')
    
    #scale to range(0,1)
    xs = (xs - np.min(xs))/np.ptp(xs)
    fs = (fs - np.min(fs))/np.ptp(fs)
    print(fs.shape)
    
    #save 
    plt.plot(xs, fs, 'gray') # Plot the samples
    #plt.title('l = sqrt(2)')
    plt.savefig('1D_'+kernel+'_'+str(l_val)+'.jpg')
    np.save('1D_'+kernel+'_'+str(l_val)+'.npy', fs)
   



if __name__ == '__main__':
    #gaussian_2D_special_case(64)
    sample_from_1D_Gaussian(64, 'rbf', 2)
