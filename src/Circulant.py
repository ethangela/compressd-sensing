from scipy.linalg import dft
from scipy.fftpack import fft, ifft
from scipy.sparse import diags
import numpy as np
from scipy import linalg

def conventional(n):
    #Fourier Matrix
    F = dft(n, 'sqrtn')

    #vector t
    t = np.random.normal(size=n)

    #Diagonal
    tmp = np.sqrt(n) * F.dot(t)
    dig = diags(tmp).toarray()

    #Circulant
    c1 = F.dot(dig)
    c = c1.dot(linalg.inv(F))
    
    return c

if __name__ == "__main__":
    rst = conventional(3)
    print(rst)