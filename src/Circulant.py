from scipy.linalg import dft
from scipy.fftpack import fft, ifft
from scipy.sparse import diags
import numpy as np
from scipy import linalg


#Scipy
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
    Ft = n * fft(t) #was np.sqrt(n)
    r2 = np.multiply(r1, Ft)
    
    #step 3: F @ Diag() @ F^{-1} @ x
    compressive = fft(r2)
    
    return compressive


#Tensorflow
def tf_fft_circulant(signal_vector, signal_size, t): 
    signal_vector = tf.convert_to_tensor(signal_vector, dtype=tf.complex64)
    t = tf.convert_to_tensor(t, dtype=tf.complex64)
    
    #signal length
    tf_n = tf.constant(signal_size, dtype=tf.complex64)
    
    #step 1: F^{-1} @ x
    r1 = tf.signal.ifft(signal_vector)
    
    #step 2: Diag() @ F^{-1} @ x
    Ft = tf_n * tf.signal.fft(t) #was tf.math.sqrt(n)
    r2 = tf.multiply(r1, Ft)
    
    #step 3: F @ Diag() @ F^{-1} @ x
    compressive = tf.signal.fft(r2)
    float_compressive = tf.cast(compressive, tf.float32)
    
    return float_compressive


#PyTorch
def torch_fft_circulant(signal_vector, signal_size, t): 
    #step 1: F^{-1} @ x
    signal_vector = signal_vector.unsqueeze(-1)
    signal_vector = torch.cat((signal_vector, torch.zeros(signal_vector.size)), -1)
    r1 = torch.ifft(signal_vector, 1)
    
    #step 2: Diag() @ F^{-1} @ x
    Ft = signal_size * torch.fft(t, 1, onesided=False) #was np.sqrt(signal_size)
    r2 = torch.mul(r1,Ft)
     
    #step 3: F @ Diag() @ F^{-1} @ x
    compressive = torch.fft(r2, 1)
    
    return compressive #NOTE: this is a complex value, not a real value, need drop the last dimension for subsequent calculations 

if __name__ == "__main__":
    signal_vector = np.array([1,2,3,4,5,6,7,8,9])
    t = np.random.normal(size=len(signal_vector))
    signal_size = 8
