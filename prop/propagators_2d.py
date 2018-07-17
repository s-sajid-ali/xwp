# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:38:01 2017

@author: sajid
"""

import numpy as np
import numexpr as ne
import pyfftw
import dask.array as da

'''
contains functions propTF, propFF, prop1FT, propIR

'''

__all__ = ['propTF',
           'prop1FT',
           'propFF',
           'propIR']


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
