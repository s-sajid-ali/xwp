#%%cython --compile-args=-fopenmp --compile-args=-O3 --link-args=-fopenmp
'''
Cythonized exact propagation in 2D.  

Transformation of the python code to cython. 

The input/output coordiantes have been collapsed to a 2D array to have
2 for loops instead of 4. This also means an index array is needed. Inner loop
is explicitly parallelized.

There is now a new add_clean function that sums all the points in an array and resets
the value of each element ot 0. This final loop ensures that even if the inner loops 
are paralleized (hopefully using something along the lines of #pragma openmp collapse),
no race conditions occur. For now, this code will be compiled with a modern gcc/intel 
compiler to benefit from auto-vectorization.

'''

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, cos, sin
from libc.stdlib cimport malloc, free
cimport openmp
cimport cython
from cython.parallel import prange

'''
This function adds the elements of the input array and resets each element to 0.
'''
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
def exact_prop_2D_cython(double complex[:,:] in_wave, double complex [:,:] out_wave,\
                        double L_in, double L_out, double x_out_off, double y_out_off,\
                         double wavel, double z, int num_threads):
    
    cdef double pi = 3.14159265359
    
    '''
    Build the input and output domains from the input
    '''
    
    cdef int N_in_x = in_wave.shape[0]
    cdef int N_in_y = in_wave.shape[1]
    cdef int N_in_t = N_in_x*N_in_y
    cdef double[:] in_x = np.linspace(-L_in/2,L_in/2,N_in_x)
    cdef double[:] in_y = np.linspace(-L_in/2,L_in/2,N_in_y)
    cdef double[:,:] in_domain  = np.dstack(np.meshgrid(in_x,in_y)).reshape(N_in_t,2)
    
    cdef int N_out_x = out_wave.shape[0]
    cdef int N_out_y = out_wave.shape[1]
    cdef int N_out_t = N_out_x*N_out_y
    cdef double[:] out_x = np.linspace(-L_out/2+x_out_off,L_out/2+x_out_off,N_out_x)
    cdef double[:] out_y = np.linspace(-L_out/2+x_out_off,L_out/2+x_out_off,N_out_y)
    cdef double[:,:] out_domain  = np.dstack(np.meshgrid(out_x,out_y)).reshape(N_out_t,2)
    
    '''
    Calculate step sizes in x and y
    '''
    cdef double step_in_x = L_in/N_in_x
    cdef double step_in_y = L_in/N_in_y
    
    cdef double complex fac1 = 0.7071067811865476-0.7071067811865476j #np.sqrt(1/1j)
    cdef double complex fac  = (step_in_x/sqrt(wavel*z))*(step_in_y/sqrt(wavel*z))*fac1
    cdef double _temp1
    
    '''
    Declare variables to be used. 
    '''
    cdef long int i,j,p,q,p1,q1
    cdef double x,x1,y,y1
    cdef double complex f
    cdef long int [:,:] indices  = np.dstack(np.meshgrid(np.arange(N_in_x),np.arange(N_in_y))).reshape(N_in_t,2)    
    
    '''
    This 1D array will hold the values that need to be summed from the inner loop.
    This is to ensure that there are no race conditions when parallelizing.
    '''
    cdef double complex *sum_temp = <double complex*> malloc(N_in_t * sizeof(double complex))
    add_clean(sum_temp,N_in_t)
    
    
    for i in range(N_out_t):
        x1 = out_domain[i][0]
        y1 = out_domain[i][1]
        p1 = indices[i][0]
        q1 = indices[i][1]
        with nogil:
            for j in prange(N_in_t, num_threads=num_threads):
                x = in_domain[j][0]
                y = in_domain[j][1]
                p = indices[j][0]
                q = indices[j][1]
                f  = in_wave[p][q]
                _temp1 = (((-1*pi)/(wavel*z))*((x-x1)**2+(y-y1)**2))
                sum_temp[j] = (cos(_temp1)+1j*sin(_temp1))*f*fac
        out_wave[p1][q1] = add_clean(sum_temp,N_in_t)
    return