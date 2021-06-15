#1D versions of propagators

import numpy as np

__all__ = ['exact_prop_numba',
           'exact_prop']


'''
Exact propagation in 1D
(Note that the function changes the values of out_wave
instead of returning an array)


in_wave   : profile of the beam at the input plane. 
out_wave  : array to be filled with values of wave at output plane
L_in      : side length of the support at input plane
L_out     : side length of the support at output plane
wavel     : wavelength
z         : the propogation distance
'''

def exact_prop(in_wave,out_wave,L_in,L_out,wavel,z):
    pi = np.pi
    N_in = np.shape(in_wave)[0]
    in_domain = np.linspace(-L_in/2,L_in/2,N_in)
    N_out = np.shape(out_wave)[0]
    out_domain = np.linspace(-L_out/2,L_out/2,N_out)
    step_in = L_in/N_in
    for i in range(N_out):
        for j in range(N_in):
            x = in_domain[j]
            f = in_wave[j]
            x1 = out_domain[i]
            out_wave[i] += f*np.exp((-1j*pi*x*x)/(wavel*z))*np.exp((-1j*2*pi*x*x1)/(wavel*z))
    out_wave *= (1/np.sqrt(1j*wavel*z))*step_in
    return

    
    
'''
Exact propagation (accelerated using numba/numexpr) in 1D
(Note that the function changes the values of out_wave
instead of returning an array)


in_wave   : profile of the beam at the input plane. 
out_wave  : array to be filled with values of wave at output plane
L_in      : side length of the support at input plane
L_out     : side length of the support at output plane
wavel     : wavelength
z         : the propogation distance
'''


try:
    from numba import jit,prange
    @jit(nopython=True, parallel=True)
    def exact_prop_numba(in_wave,out_wave,L_in,L_out,wavel,z):
        pi = np.pi
        N_in  = in_wave.shape[0]
        N_out = out_wave.shape[0]
        in_domain  = np.linspace(-L_in/2,L_in/2,N_in)
        out_domain = np.linspace(-L_out/2,L_out/2,N_out)
        step_in = L_in/N_in
        for i in prange(N_out):
            for j in prange(N_in):
                x = in_domain[j]
                f = in_wave[j]
                x1 = out_domain[i]
                out_wave[i] += f*np.exp((-1j*pi*(x-x1)**2)/(wavel*z))*(step_in/np.sqrt(1j*wavel*z))
        return
except:
    pass
