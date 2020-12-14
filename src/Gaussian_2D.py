import numpy as np
#from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
#from matplotlib import cm
from argparse import ArgumentParser

def kernel(x1, x2, ker, l=3, sigma_f=1):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    d = np.sqrt(dist_matrix)
    if ker == 'rbf':
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        conv = sigma_f ** 2 * np.exp(-1 / (l**2) * dist_matrix)
    elif ker == 'exp':
        conv = np.exp(-1 / l * d)
    elif ker == 'per':
        conv = np.exp(-2 / (l**2) * np.sin(d/2)**2)
    return conv

def sample_from_2D_Gaussian(size, ker, l, idx):
    m = size
    test_d1 = np.linspace(-10, 10, m)
    test_d2 = np.linspace(-10, 10, m)
    
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]
    test_X = np.asarray(test_X)
    mu = np.zeros_like(test_d1)
    cov = kernel(test_X, test_X, ker, l)
    print('parameter set done')
    
    #gp_samples = np.random.multivariate_normal(mean=mu.ravel(), cov = cov, size = 1)
    gp_samples = multivariate_normal(mean=mu.ravel(), cov=cov, allow_singular =True).rvs(1).T #(m,)
    z = gp_samples.reshape(test_d1.shape)
    print('z shape is{}'.format(z.shape))
    print('sampling done')
    
    #scale to range(0,1)
    z = (z - np.min(z))/np.ptp(z)
    name = 'image/Gaussian signal/2D_' + ker +'_'+ str(l) +'_'+ str(size) +'_'+ str(idx) + '.npy'
    np.save(name, z)
    #print(z)
    
    ##test_d1 = (test_d1 - np.min(test_d1))/np.ptp(test_d1)
    ##test_d2 = (test_d2 - np.min(test_d2))/np.ptp(test_d2)
    ##print('scaling done')
    ##fig = plt.figure(figsize=(5, 5))
    ##plt.contourf(test_d1, test_d2, z, cmap=cm.coolwarm, alpha=1)
    ##img_name = '2D_' + ker + '_' + str(l) + '.jpg'
    ##plt.savefig(img_name)
    
    
def main(hparams):
    return sample_from_2D_Gaussian(hparams.size, hparams.kernel, hparams.length, 1) 


if __name__ == '__main__':

    PARSER = ArgumentParser()
    PARSER.add_argument('--size', type=int, default=256, help='size')
    PARSER.add_argument('--kernel', type=str, default='rbf', help='kernel')
    PARSER.add_argument('--length', type=float, default=3.0, help='length')
    
    HPARAMS = PARSER.parse_args()
    
    main(HPARAMS)