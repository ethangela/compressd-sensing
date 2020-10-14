from scipy.linalg import dft
from scipy.fft import fft, ifft
from scipy.sparse import diags
import numpy as np
from scipy import linalg
from scipy.linalg import norm


#Scipy
def conventional_circulant(signal_vector, t):
    #signal length
    n = len(signal_vector)
    
    #Fourier Matrix
    F = dft(n, scale='sqrtn') 

    #Diagonal
    dig = diags(np.sqrt(n) * F.dot(t)).toarray()

    #Circulant
    c1 = F.dot(dig)
    ct = c1.dot(linalg.inv(F))
    
    #Associate with signal
    compressive = ct.dot(signal_vector)
    
    return compressive
  

def fft_circulant(signal_vector, t, test=False): 
    #signal length
    n = len(signal_vector)
    
    #step 1: F^{-1} @ x
    if not test:
        r1 = ifft(signal_vector, norm="ortho")
    else:
        r1 = ifft(signal_vector)
    #step 2: Diag() @ F^{-1} @ x
    if not test: 
        Ft = np.sqrt(n) * fft(t, norm="ortho")
    else:
        Ft = fft(t)
    r2 = np.multiply(r1, Ft)
    
    #step 3: F @ Diag() @ F^{-1} @ x
    if not test:
        compressive = fft(r2, norm="ortho")
    else:   
        compressive = fft(r2)
    
    return compressive


#Tensorflow
def circulant_tf(signal_vector, t, signal_size):  
    signal_vector = tf.convert_to_tensor(signal_vector, dtype=tf.complex64)
    t = tf.convert_to_tensor(t, dtype=tf.complex64)
    
    #step 1: F^{-1} @ x
    r1 = tf.signal.ifft(signal_vector, name='circulant_step1_ifft')               
    
    #step 2: Diag() @ F^{-1} @ x
    Ft = tf.signal.fft(t)
    r2 = tf.multiply(r1, Ft, name='circulant_step2_diag')                
    
    #step 3: F @ Diag() @ F^{-1} @ x
    compressive = tf.signal.fft(r2, name='circulant_step3_fft')
                         
    with tf.Session() as sess:
        print(sess.run(compressive))
    sess.close()



if __name__ == "__main__":
    import tensorflow as tf
    
    signal_vector = np.array([1,2,3,4,5,6,7,8,9,10])
    t = np.random.normal(size=len(signal_vector))   
    r1 = conventional_circulant(signal_vector, t)
    r2 = fft_circulant(signal_vector, t)
    r3 = fft_circulant(signal_vector, t, test=True)
    
    print(r1)
    print('\t')
    print(r2)
    print('\t')
    print(r3)
    print('\t')
    circulant_tf(signal_vector,t,10)
    
    