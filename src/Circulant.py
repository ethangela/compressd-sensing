from scipy.linalg import dft
from scipy.fftpack import fft, ifft
from scipy.sparse import diags
import numpy as np
from scipy import linalg



def conventional_circulant(signal_vector, t):
    #signal length
    n = len(signal_vector)
    
    #Fourier Matrix
    F = dft(n) #should be with 'sqrtn', however we choose to ignore the normlisation in order to align with fft()

    #Diagonal
    dig = diags(np.sqrt(n) * F.dot(t)).toarray()

    #Circulant
    c1 = F.dot(dig)
    ct = c1.dot(linalg.inv(F))
    
    #Associate with signal
    compressive = ct.dot(signal_vector)
    
    return compressive
  

  
def fft_circulant(signal_vector, t): 
    #signal length
    n = len(signal_vector)
    
    #step 1: F^{-1} @ x
    r1 = ifft(signal_vector)
    
    #step 2: Diag() @ F^{-1} @ x
    Ft = np.sqrt(n) * fft(t)
    r2 = np.multiply(r1, Ft)
    
    #step 3: F @ Diag() @ F^{-1} @ x
    compressive = fft(r2)
    
    return compressive

if __name__ == "__main__":
    signal_vector = np.array([1,2,3,4,5,6,7,8,9])
    t = np.random.normal(size=len(signal_vector))   
    r1 = conventional_circulant(signal_vector, t)
    r2 = fft_circulant(signal_vector, t)
    
    print(r1)
    print('\t')
    print(r2)