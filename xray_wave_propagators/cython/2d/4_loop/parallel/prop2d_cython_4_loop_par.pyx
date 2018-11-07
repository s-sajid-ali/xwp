##%%cython --compile-args=-fopenmp --compile-args=-O3 --link-args=-fopenmp
'''
Cythonized exact propagation in 2D.  

Transformation of the python code to cython. 

Only difference being that there is now a new add_clean function that sums all the 
points in an array and resets the value of each element ot 0. This final loop ensures 
that even if the inner loops are paralleized (hopefully using something along the 
lines of #pragma openmp collapse), no race conditions occur. For now, this code will
be compiled with a modern gcc/intel compiler to benefit from auto-vectorization.

'''

import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp
from libc.math cimport sqrt, cos, sin
from libc.stdlib cimport malloc, free
from cython.parallel import prange,threadid,parallel

'''
This function adds the elements of the input array and resets each element to 0. 
Using 1 thread, set all elements of the sum_temp array to 0.
Use multple threads to perform addition.
Using 1 thread, combine the partial sums.
'''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex add_clean(double complex * arr, int N):
    
    cdef int sum_threads = 30
    cdef double complex *sum_temp = <double complex*> malloc(sum_threads * sizeof(double complex))
    cdef int k,thread_id
    cdef double complex out = 0+0j
    cdef int offset = int(N/sum_threads)
    
    for k in range(sum_threads):
        sum_temp[k] = 0
        
    with nogil, parallel(num_threads = sum_threads):
        thread_id = threadid()
        for k in range(offset):
            sum_temp[thread_id] += arr[k + offset*thread_id]
            arr[k + offset*thread_id] = 0    

    for k in range(sum_threads):
        out+=sum_temp[k]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def exact_prop_2D_cython(double complex [:,:] in_wave, double complex [:,:] out_wave,\
                        double L_in, double L_out, double x_out_off, double y_out_off,\
                         double wavel, double z):
    cdef double pi = 3.14159265359
    
    '''
    Build the input and output domains from the input
    '''
    cdef int N_in_x = in_wave.shape[0]
    cdef int N_in_y = in_wave.shape[1]
    cdef int N_in_t = N_in_x*N_in_y
    cdef double[:] in_x = np.linspace(-L_in/2,L_in/2,N_in_x)
    cdef double[:] in_y = np.linspace(-L_in/2,L_in/2,N_in_y)
    
    cdef int N_out_x = out_wave.shape[0]
    cdef int N_out_y = out_wave.shape[1]
    cdef double[:] out_x = np.linspace(-L_out/2+x_out_off,L_out/2+x_out_off,N_out_x)
    cdef double[:] out_y = np.linspace(-L_out/2+x_out_off,L_out/2+x_out_off,N_out_y)
    
    '''
    Calculate step sizes in x and y
    '''
    cdef double step_in_x = L_in/N_in_x
    cdef double step_in_y = L_in/N_in_y
    
    '''
    Declare variables to be used. 
    '''
    cdef double x,x1,y,y1
    cdef double complex f
    cdef int i,j,k,p,q
    
    cdef double complex fac1 = 0.7071067811865476-0.7071067811865476j #np.sqrt(1/1j)
    cdef double complex fac  = (step_in_x/sqrt(wavel*z))*(step_in_y/sqrt(wavel*z))*fac1
    
    
    '''
    This 1D array will hold the values that need to be summed from the inner loop.
    This is to ensure that there are no race conditions when parallelizing.
    '''
    cdef double complex *sum_temp = <double complex*> malloc(N_in_t * sizeof(double complex))
    add_clean(sum_temp,N_in_t)
    
    
    
    for i in range(N_out_x):
        for j in range(N_out_y):
            with nogil:
                for p in prange(N_in_x,num_threads=5):
                    for q in prange(N_in_y,num_threads=5):
                        x  =  in_x[p]
                        y  =  in_y[q]
                        x1 = out_x[i]
                        y1 = out_y[j]
                        f  = in_wave[p][q]
                        k  = p*N_in_x + q 
                        sum_temp[k] = (cos((((-1*pi)/(wavel*z))*((x-x1)**2+(y-y1)**2)))+1j*sin((((-1*pi)/(wavel*z))*((x-x1)**2+(y-y1)**2))))*f*fac
            out_wave[i][j] = add_clean(sum_temp,N_in_t)
    return