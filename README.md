## xray_wave_propagators
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

- via custom install script (imported as binaries):  
  - Direct evaluation of fresnel integral in 1D and 2D in cython. 
 
- in jupyter notebooks :
  - Finite Difference propagation in 2D.
  - Wigner Distribution Function for 1D signals.

#### In Progress :
- Fractional Fourier Transform based wave propagation.

These propagators are tested for various regimes to glean their limits of validity.
