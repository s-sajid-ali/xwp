## xray-wave-propagators
Code for evaluation of wave propagation using Fresnel/Fraunhofer approximation. 

#### Currently implemented :
- Direct evaluation of fresnel integral in 1D and 2D. Optimzied in cython/numba. 
- spectral appromximations of fresnel integral:
  - Transfer function
  - Impulse Response
  - Single Fourier Transform
- Fraunhofer integral
- 1D Finite Difference propagators

#### In Progress :
- 2D Finite Difference propagators
- Fractional Fourier Transform based wave propagation.

These propagators are tested for various regimes to glean their limits of validity.
