##%%cython --compile-args=-fopenmp --compile-args=-O3 --link-args=-fopenmp --force 
'''
Cythonized exact propagation in 1D.  
'''
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, cos, sin
from libc.stdlib cimport malloc, free
cimport cython
cimport openmp
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex add_clean(double complex *arr, int N):
    cdef double complex out = 0+0j
    cdef int k
    for k in range(N):
        out+=arr[k]
        arr[k] = 0
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef exact_prop_cython(double[:] in_wave, double complex [:] out_wave,\
               double L_in, double L_out, double wavel, double z):
    
    cdef double pi = 3.14159265359
    cdef int N_in  = in_wave.shape[0]
    cdef int N_out = out_wave.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] in_domain  = np.linspace(-L_in/2,L_in/2,N_in)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_domain = np.linspace(-L_out/2,L_out/2,N_out)
    cdef double step_in = L_in/N_in
    
    cdef double complex fac1 = 0.7071067811865476-0.7071067811865476j #np.sqrt(1/1j)
    cdef double complex fac  = (step_in/sqrt(wavel*z))*fac1
    cdef double x,x1,f
    cdef int i,j
    cdef double _temp1
    cdef double complex _temp2
    
    cdef double complex *sum_temp = <double complex*> malloc(N_in * sizeof(double complex))
    add_clean(sum_temp,N_in)
    
    for i in range(N_out):
        x1 = out_domain[i]
        with nogil:
            for j in prange(N_in, num_threads = 10):
                f =   in_wave[j]
                x = in_domain[j]
                _temp1 = (((-1*pi)/(wavel*z))*(x-x1)**2)
                _temp2 = cos(_temp1)+1j*sin(_temp1)
                sum_temp[j] = _temp2*f*fac
        out_wave[i] = add_clean(sum_temp,N_in)
    return