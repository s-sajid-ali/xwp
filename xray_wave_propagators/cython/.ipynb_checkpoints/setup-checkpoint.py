from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os 

setup(
    
    ext_modules=[
        Extension('prop1d_cython',
        sources=['prop1d_cython.pyx'],
        extra_compile_args=['-O3','-fopenmp','-xcore-avx2'],
        language='c')
        ],
    
    include_dirs = [np.get_include()],
    
    cmdclass = {'build_ext': build_ext}
    
)