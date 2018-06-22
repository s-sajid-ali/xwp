'''
pyFFT objects for 2D fft evaluation. Wisdom stores the plans making evaluation faster.
'''

import pyfftw
import numpy as np
import os,pickle
from os.path import dirname as up


try:
    pyfftw.import_wisdom(pickle.load(open(up(os.getcwd())+'/wisdom/wisdom.pickle','rb')))
    print('Wisdom loaded!')
except:
    pass


__all__ = ['FFT_2d_Obj']

'''
pyfftw builder interface is used to access the FFTW class and is optimized with pyfftw_wisdom. Multithreading is used to speed up the calculation. The input is destroyed as making a copy of the input is time consuming and is not useful.

The advantage of using a class is the reduction of overhead associated with creating it every time it is used in a loop.

TODO : A rule of thumb to set the number of threads given the size of transform. 
'''

class FFT_2d_Obj(object):
        
    def __init__(self,dimension,direction='FORWARD',flag='ESTIMATE',threads = 24):
        
        self.pyfftw_array = pyfftw.empty_aligned(dimension,dtype='complex128', n = pyfftw.simd_alignment)
        
        self.fft2_ = pyfftw.FFTW(self.pyfftw_array,self.pyfftw_array, axes=(0,1), direction='FFTW_FORWARD',
                                 flags=('FFTW_'+str(flag), ), threads=threads, planning_timelimit=None )
        self.ifft2_ = pyfftw.FFTW(self.pyfftw_array,self.pyfftw_array, axes=(0,1), direction='FFTW_BACKWARD',
                                  flags=('FFTW_'+str(flag), ), threads=threads, planning_timelimit=None)
        
        self.ifft2_.__call__(normalise_idft='False')
        
    def run_fft2(self,A):
        pa = self.pyfftw_array
        np.copyto(pa,A)
        self.fft2_()
        np.copyto(A,pa)
        del(pa)
        return None
    
    def run_ifft2(self,A):
        pa = self.pyfftw_array
        np.copyto(pa,A)
        self.ifft2_()
        np.copyto(A,pa)
        del(pa)
        return None