## xwp : X-ray Wave Propagators
Code for evaluation of wave propagation. Tools for visualization of signals in phase-space.

#### Currently implemented  :
- as part of the package :
  - Direct evaluation of fresnel integral in 1D and 2D. Optimzied in numba. 
  - spectral appromximations of fresnel integral:
    - Transfer function
    - Impulse Response
    - Single Fourier Transform
  - Fraunhofer integral
  - Finite Difference propagation in 1D.

- as part of the related xwp_cython package:  
  - Direct evaluation of fresnel integral in 1D and 2D in cython. 
 
- in jupyter notebooks :
  - Finite Difference propagation in 2D.
  - Wigner Distribution Function for 1D signals.

#### In Progress :
- Fractional Fourier Transform based wave propagation.


#### Note:
All physical quantities have SI units. 
The implementation of finite difference wave propagation is just for proof of concept purposes and is not optimized (or parallelized). 
The direct wave propagation is somewhat optimized but much more can be done in terms of memeory reuse (so that the same part of the input wave is not repeatedly flushed from the cache).
