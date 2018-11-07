#2D versions of propagators

import numpy as np
import numexpr as ne
import pyfftw
from numba import jit,prange
import dask.array as da

'''
contains functions propTF, propFF, prop1FT, propIR

'''

__all__ = ['propTF',
           'prop1FT',
           'propFF',
           'propIR',
           'exact_prop',
           'exact_prop_numba']

'''
Propogation using the Transfer function method.

Inputs - 
u          : profile of the beam at the input plane. 
step       : is the sampling step size at the input plane.
L1         : side length of the support.
wavel      : the wavelength of the light
z          : the propogation distance
fft_object : 
Outputs -
u     : beam profile at the output plane
L1    : the side length of the support at the output plane.

'''

def propTF(u,step,L1,wavel,z,fft_object = None) :
    M,N = np.shape(u)
    pi = np.pi
    FX,FY = da.meshgrid(np.fft.fftfreq(M,step),np.fft.fftfreq(N,step))
    
    if fft_object != None :
        fft_object.run_fft2(u)
    else :
        u = np.fft.fft2(u)
    
    u = ne.evaluate('exp(-1j*(2*pi*z/wavel)*sqrt(1-wavel**2*(FX**2+FY**2)))*u')
    
    if fft_object != None :
        fft_object.run_ifft2(u)
    else: 
        u = np.fft.ifft2(u)
    
    return u,L1

'''
Propogation using the Single Fourier Transform approach. Input convention as above.

Inputs - 
u     : profile of the beam at the input plane. 
step  : is the sampling step size at the input plane.
L1    : side length of the support.
wavel : the wavelength of the light
z     :the propogation distance

Outputs - 
u     : beam profile at the output plane
L_out : the side length of the support at the output plane.

'''
def prop1FT(u,step,L1,wavel,z,fft_object = None):
    M,N = np.shape(u)
    k = 2*np.pi/wavel
    x = np.linspace(-L1/2.0,L1/2.0-step,M)
    y = np.linspace(-L1/2.0,L1/2.0-step,N)
    
    '''
    #Kenan's approach
    fx = np.fft.fftfreq(M,d=step)
    fy = np.fft.fftfreq(N,d=step)
    fx = pyfftw.interfaces.numpy_fft.fftshift(fx)
    fy = pyfftw.interfaces.numpy_fft.fftshift(fy)
    FX,FY = da.meshgrid((fx),(fy))
    c = np.exp((-1j*z*2*np.pi/wavel)*np.sqrt(1+wavel**2*(FX**2+FY**2)))
    '''
    
    L_out = wavel*z/step
    step2 = wavel*z/L1
    pi = np.pi
    X,Y = da.meshgrid(x,y)
    u  = ne.evaluate('exp((-1j*2*pi/wavel)*sqrt(X**2+Y**2+z**2))*u')
    del X,Y
    
    if fft_object != None :
        fft_object.run_fft2(u)
    else:
        u = np.fft.fft2(u)
    
    u = np.fft.fftshift(u)
    
    x2 = np.linspace(-L_out/2.0,L_out/2.0,M)
    y2 = np.linspace(-L_out/2.0,L_out/2.0,N)
    X2,Y2 = da.meshgrid(x2,y2)
    u   = ne.evaluate('exp((-1j*2*pi/wavel)*sqrt(X2**2+Y2**2+z**2))*u')
    del X2,Y2
    
    u = ne.evaluate('u*(1j/(wavel*z))*step*step')
    
    return u,L_out



'''
Fraunhofer propogation. 

Inputs -
u     : profile of the beam at the input plane. 
step  : is the sampling step size at the input plane.
L1    : side length of the support.
wavel : the wavelength of the light
z     :the propogation distance

Outputs -
u     : beam profile at the output plane
L_out : the side length of the support at the output plane.
'''
def propFF(u,step,L1,wavel,z,fft_object = None):
    M,N = np.shape(u)
    k = 2*np.pi/wavel
    L_out = wavel*z/step
    step2 = wavel*z/L1
    n = M #number of samples
    x2 = np.linspace(-L_out/2.0,L_out/2.0,M)
    y2 = np.linspace(-L_out/2.0,L_out/2.0,N)
    X2,Y2 = np.meshgrid(x2,y2) 
    
    c =ne.evaluate('exp((1j*k*(1/(2*z)))*(X2**2+Y2**2))')*(1/(1j*wavel*z))
    u = pyfftw.interfaces.numpy_fft.fftshift(u)
    
    if fft_object != None :
        fft_object.run_fft2(u)
    else:
        u = np.fft.fft2(u)
    
    u = pyfftw.interfaces.numpy_fft.ifftshift(u)
    u = ne.evaluate('c*u')
    u *= step*step
    
    return u,L_out



'''
Warning : use is now Deprecated !
Propogation using the Impulse Response function. The convention of shiftinng a function in realspace before performing the fourier transform which is used in the reference is followed here. Input convention as above. Use is deprecated since the implementation of 1FT for ranges that are too large for TF but too small for FF. 
'''
def propIR(u,step,L,wavel,z,fft_object = None):
    M,N = np.shape(u)
    k = 2*np.pi/wavel
    x = np.linspace(-L/2.0,L/2.0-step,M)
    y = np.linspace(-L/2.0,L/2.0-step,N)
    X,Y = np.meshgrid(x,y)
    
    h = ne.evaluate('(exp(1j*k*z)/(1j*wavel*z))*exp(1j*k*(1/(2*z))*(X**2+Y**2))')
    h_in = pyfftw.empty_aligned((np.shape(h)))
    h = pyfftw.interfaces.numpy_fft.fftshift(h)
    h_in = h
    
    if fft_object != None :
        fft_object.run_fft2(h)
    else:
        h = np.fft.fft2(h)
    
    H = h*step*step
    
    u =np.fft.fftshift(u)
    
    if fft_object != None :
        fft_object.run_fft2(u)
    u = np.fft.fft2(u)
    
    u = ne.evaluate('H * u')

    if fft_object != None :
        fft_object.run_ifft2(u)
    else:
        u = np.fft.ifft2(u)
    
    u = np.fft.ifftshift(u)
    
    return u


'''
Exact propagation in 2D. 

First attempt at getting the logic correctly, optimized using cython/numba later.

Vectorized by performing the numerical integral at each output point using numexpr
over the whole input array.
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
Exact propagation in 2D using numba by adding the @jit decorator & prange.

Gives some speedup but it generally brittle to 
explicit parallelization though it works here.

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


