#1D versions of propagators

import numpy as np
import numexpr as ne

'''
Exact propagation in 1D. 
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
            #out_wave[i] += f*np.exp((-1j*pi*x*x)/(wavel*z))*np.exp((-1j*2*pi*x*x1)/(wavel*z))
            out_wave[i] += ne.evaluate('f*exp((-1j*pi*x*x)/(wavel*z))*exp((-1j*2*pi*x*x1)/(wavel*z))')
    out_wave *= (1/np.sqrt(1j*wavel*z))*step_in
    return


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
    N = np.shape(u)[0]
    pi = np.pi
    F = np.fft.fftfreq(N,step)
    
    u = np.fft.fft(u)
    u = ne.evaluate('exp(-1j*(2*pi*z/wavel)*sqrt(1-wavel**2*(F**2)))*u')
    u = np.fft.ifft(u)
    
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
    N = np.shape(u)[0]
    k = 2*np.pi/wavel
    x = np.linspace(-L1/2.0,L1/2.0,N)
    L_out = wavel*z/step
    step2 = wavel*z/L1
    pi = np.pi
    
    
    #Kenan's approach
    f = np.fft.fftfreq(N,d=step)
    f = np.fft.fftshift(f)
    #c = np.exp((-1j*z*2*np.pi/wavel)*np.sqrt(1+wavel**2*(f**2)))
    #c = np.exp((-1j*2*np.pi/wavel)*np.sqrt(x**2+z**2))
    #u = u*c
    
    u  = ne.evaluate('exp(1j*pi/(wavel*z)*(x**2))*u')
        
    u = np.fft.fft(u)*step
    u = np.fft.fftshift(u)
    
    #x2 = np.linspace(-L_out/2.0,L_out/2.0,N)
    #u  = ne.evaluate('exp((-1j*2*pi/wavel)*sqrt(x2**2+z**2))*u')
    u = ne.evaluate('u*(sqrt(1/(1j*wavel*z)))')
    
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
    N = np.shape(u)[0]
    k = 2*np.pi/wavel
    L_out = wavel*z/step
    step2 = wavel*z/L1
    n = N #number of samples
    x2 = np.linspace(-L_out/2.0,L_out/2.0,N)

    #c = ne.evaluate('exp(((1j*k)/(2*z))*(x2**2))')
    
    u = np.fft.fft(u)*step
    u = np.fft.fftshift(u)
    
    #u = ne.evaluate('c*u')
    u = u*np.sqrt(1/(1j*wavel*z))
    
    return u,L_out



'''
Warning : use is now Deprecated !
Propogation using the Impulse Response function. The convention of shiftinng a function in realspace before performing the fourier transform which is used in the reference is followed here. Input convention as above. Use is deprecated since the implementation of 1FT for ranges that are too large for TF but too small for FF. 
'''
def propIR(u,step,L,wavel,z,fft_object = None):
    N = np.shape(u)[0]
    k = 2*np.pi/wavel
    x = np.linspace(-L/2.0,L/2.0,N)
    
    #h = ne.evaluate('(exp(1j*k*z)/(1j*wavel*z))*exp(((1j*k)/(2*z))*(x**2))')
    #h = np.exp(1j*k*z)*np.exp(((1j*k)/(2*z))*(x**2))
    h = np.sqrt(1/(1j*wavel*z))*np.exp(((1j*k)/(2*z))*(x**2))

    h = np.fft.fft(np.fft.fftshift(h))*step

    u = np.fft.fft(u)
    u *= h
    #u = ne.evaluate('h * u')
    u = np.fft.ifft(u)
    
    return u,L