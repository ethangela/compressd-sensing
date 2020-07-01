import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm

def kernel(x1, x2, ker, l=2, sigma_f=1):#################################################
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

def sample_from_2D_Gaussian(size, ker):
    m = size
    test_d1 = np.linspace(-10, 10, m)
    test_d2 = np.linspace(-10, 10, m)
    
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]
    test_X = np.asarray(test_X)
    mu = np.zeros_like(test_d1)
    cov = kernel(test_X, test_X, ker)
    print('parameter set done')
    
    #gp_samples = np.random.multivariate_normal(mean=mu.ravel(), cov = cov, size = 1)
    gp_samples = multivariate_normal(mean=mu.ravel(), cov=cov, allow_singular =True).rvs(1).T #(m,)
    z = gp_samples.reshape(test_d1.shape)
    print('z shape is{}'.format(z.shape))
    print('sampling done')
    
    #scale to range(0,1)
    z = (z - np.min(z))/np.ptp(z)
    name = '2D_' + ker + '.npy'
    np.save(name, z)
    #print(z)
    test_d1 = (test_d1 - np.min(test_d1))/np.ptp(test_d1)
    test_d2 = (test_d2 - np.min(test_d2))/np.ptp(test_d2)
    print('scaling done')
    
    fig = plt.figure(figsize=(5, 5))
    plt.contourf(test_d1, test_d2, z, cmap=cm.coolwarm, alpha=1)
    #ax.set_title("with optimization l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    #plt.show()
    img_name = '2D_' + ker + '.jpg'
    plt.savefig(img_name)
    
    
if __name__ == '__main__':    
    sample_from_2D_Gaussian(64, 'exp')