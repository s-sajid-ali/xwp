#2D versions of propagators

import numpy as np
import numexpr as ne
import pyfftw
from numba import jit,prange
import dask.array as da


__all__ = ['exact_prop',
           'exact_prop_numba']


'''
Exact propagation in 2D. 

First attempt at getting the logic correctly,
optimized using cython/numba later.

Vectorized by performing the numerical integral at
each output point using numexpr over the whole input array.

(Note that the function changes the values of 
out_wave instead of returning an array)


in_wave   : profile of the beam at the input plane. 
out_wave  : array to be filled with values of wave at output plane
L_in      : side length of the support at input plane
L_out     : side length of the support at output plane
wavel     : wavelength
z         : the propogation distance

'''
def exact_prop(in_wave,out_wave,L_in,L_out,wavel,z):
    pi = np.pi
    
    '''
    Build the input and output domains from the input
    '''
    N_in_x = np.shape(in_wave)[0]
    N_in_y = np.shape(in_wave)[1]
    in_domain_x = np.linspace(-L_in/2,L_in/2,N_in_x)
    in_domain_y = np.linspace(-L_in/2,L_in/2,N_in_y)
    
    
    N_out_x = np.shape(out_wave)[0]
    N_out_y = np.shape(out_wave)[1]
    out_domain_x = np.linspace(-L_out/2,L_out/2,N_out_x)
    out_domain_y = np.linspace(-L_out/2,L_out/2,N_out_y)
    
    step_in_x = L_in/N_in_x
    step_in_y = L_in/N_in_y
    X_in,Y_in = np.meshgrid(in_domain_x,in_domain_y)
    '''
    Outer loops over i,j -> loop over output array
    Inner loops over p,q -> loop over input array
    For each ouput point, calculate the contribution from each input point and sum 
    '''
    fac = ((-1j*pi)/(wavel*z))
    for i in range(N_out_x):
        for j in range(N_out_y):
            x1 = out_domain_x[i]
            y1 = out_domain_y[j]
            out_wave[i][j] = ne.evaluate('sum(in_wave*exp(fac*((X_in-x1)**2+(Y_in-y1)**2)))')
    '''
    Finally scale the output
    '''
    out_wave *= ((1/np.sqrt(1j*wavel*z))*step_in_x)*((1/np.sqrt(1j*wavel*z))*step_in_y)
    return


'''
Exact propagation in 2D using numba by 
adding the @jit decorator & prange.

Gives some speedup but it generally brittle to 
explicit parallelization though it works here.


(Note that the function changes the values of
out_wave instead of returning an array)


in_wave   : profile of the beam at the input plane. 
out_wave  : array to be filled with values of wave at output plane
L_in      : side length of the support at input plane
L_out     : side length of the support at output plane
wavel     : wavelength
z         : the propogation distance
'''
@jit(nopython=True, parallel=True)
def exact_prop_numba(in_wave,out_wave,L_in,L_out,wavel,z):
    pi = np.pi
    '''
    Build the input and output domains from the input
    '''
    N_in_x = in_wave.shape[0]
    N_in_y = in_wave.shape[1]
    in_domain_x = np.linspace(-L_in/2,L_in/2,N_in_x)
    in_domain_y = np.linspace(-L_in/2,L_in/2,N_in_y)
    
    N_out_x = out_wave.shape[0]
    N_out_y = out_wave.shape[1]
    out_domain_x = np.linspace(-L_out/2,L_out/2,N_out_x)
    out_domain_y = np.linspace(-L_out/2,L_out/2,N_out_y)
    
    step_in_x = L_in/N_in_x
    step_in_y = L_in/N_in_y
    
    '''
    Outer loops over i,j -> loop over output array
    Inner loops over p,q -> loop over input array
    For each ouput point, calculate the contribution from each input point and sum 
    '''
    for i in prange(N_out_x):
        for j in prange(N_out_y):
            for p in range(N_in_x):
                for q in range(N_in_y):
                    x  =  in_domain_x[p]
                    y  =  in_domain_y[q]
                    x1 = out_domain_x[i]
                    y1 = out_domain_y[j]
                    f  = in_wave[p][q]
                    out_wave[i][j] += f*np.exp(((-1j*pi)/(wavel*z))*((x-x1)**2+(y-y1)**2))
    out_wave *= ((1/np.sqrt(1j*wavel*z))*step_in_x)*((1/np.sqrt(1j*wavel*z))*step_in_y)
    return


