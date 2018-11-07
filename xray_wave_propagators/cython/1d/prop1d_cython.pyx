##%%cython --compile-args=-fopenmp --compile-args=-O3 --link-args=-fopenmp --force 
'''
Cythonized exact propagation in 1D. Parallelized inner loop.
'''
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, cos, sin
from libc.stdlib cimport malloc, free
cimport cython
cimport openmp
from cython.parallel import prange,threadid,parallel

'''
This function adds the elements of the input array and resets each element to 0. 
Using 1 thread, set all elements of the result_temp array to 0.
Use multple threads to perform addition.
Using 1 thread, combine the partial sums.
'''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add_clean(double complex* arr, double complex* result_temp,\
                    double complex* result, int N, int result_threads):
    result[0] = 0+0j
    cdef int k,thread_id
    cdef int offset = int(N/result_threads)
    
    for k in range(result_threads):
        result_temp[k] = 0
        
    with nogil, parallel(num_threads = result_threads):
        thread_id = threadid()
        for k in range(offset):
            result_temp[thread_id] += arr[k + offset*thread_id]
            arr[k + offset*thread_id] = 0    

    for k in range(result_threads):
        result[0] += result_temp[k]
    
    return 
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef exact_prop_cython(double complex [:] in_wave, double complex [:] out_wave,\
               double L_in, double L_out, double wavel, double z):
    
    cdef double pi = 3.14159265359
    
    '''
    Build the input and output domains from the input
    '''
    cdef int N_in  = in_wave.shape[0]
    cdef int N_out = out_wave.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] in_domain  = np.linspace(-L_in/2,L_in/2,N_in)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_domain = np.linspace(-L_out/2,L_out/2,N_out)
    
    '''
    Calculate step size of input
    '''
    cdef double step_in = L_in/N_in
    
    cdef double complex fac1 = 0.7071067811865476-0.7071067811865476j #np.sqrt(1/1j)
    cdef double complex fac  = (step_in/sqrt(wavel*z))*fac1
    
    '''
    Declare variables to be used. 
    '''
    cdef double x,x1
    cdef double complex f
    cdef int i,j
    cdef double _temp1
    
    '''
    sum_temp    : this 1D array will hold the values that need to be summed from the inner loop.
    res_temp    : this holds the partial sum as the values of sum_temp are added in parallel.
    res_threads : number of threads used to sum the result
    res         : temporary varible to flush the sum_temp 
    '''
    cdef double complex *sum_temp = <double complex*> malloc(N_in * sizeof(double complex))
    
    cdef int res_threads
    res_threads = 30
    cdef double complex *res_temp = <double complex*> malloc(res_threads * sizeof(double complex))
    cdef double complex res 
    res  = 0+0j
    add_clean(sum_temp, res_temp, &res, N_in, res_threads)
    
    for i in range(N_out):
        x1 = out_domain[i]
        with nogil:
            for j in prange(N_in, num_threads = 10):
                f =   in_wave[j]
                x = in_domain[j]
                _temp1 = (((-1*pi)/(wavel*z))*(x-x1)**2)
                sum_temp[j] = (cos(_temp1)+1j*sin(_temp1))*f*fac
        add_clean(sum_temp, res_temp, &out_wave[i], N_in, res_threads)
    return