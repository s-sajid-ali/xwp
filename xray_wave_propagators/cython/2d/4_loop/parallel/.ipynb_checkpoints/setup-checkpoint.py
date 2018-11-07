from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    
    ext_modules=[
        Extension('prop2d_cython_4_loop_par',
        sources=['prop2d_cython_4_loop_par.pyx'],
        extra_compile_args=['-O3','-fopenmp','-xcore-avx2'],
        language='c')
        ],
    
    include_dirs = [np.get_include()],
    
    cmdclass = {'build_ext': build_ext}
    
)