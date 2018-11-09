#Finite Difference based wave propagation in 1D

import numpy as np
from scipy.sparse import diags
import scipy.sparse.linalg as splinalg
from tqdm import trange

all = ['finite_diff_1d_free_space',
      'finite_diff_1d_matter']


def finite_diff_1d_free_space(dim_x,dim_z,r_z,wave,wave_history):
    for i in trange(dim_x):
        c = np.ones(dim_z-2)*(1-r_z)
        d = c*wave[1:-1] + 0.5*r_z*wave[2:] + 0.5*r_z*wave[:-2]
        d[0]  += r_z*wave[0]
        d[-1] += r_z*wave[-1]
        b_diag = np.ones(dim_z-2)*(1 + r_z)
        B = diags(b_diag,offsets=0) +\
            diags(-r_z/2*np.ones(dim_z-3),offsets=1) +\
            diags(-r_z/2*np.ones(dim_z-3),offsets=-1)
        wave[1:-1] = splinalg.spsolve(B,d)
        wave_history[1:-1,i] = wave[1:-1]


def finite_diff_1d_matter(dim_x,dim_z,r_z,C,wave,wave_history):
    for i in trange(dim_x):
        c = C[1:-1,i] + np.ones(dim_z-2)*(1-r_z)
        d = c*wave[1:-1] + 0.5*r_z*wave[2:] + 0.5*r_z*wave[:-2]
        d[0]  += r_z*wave[0]
        d[-1] += r_z*wave[-1]
        b_diag = np.ones(dim_z-2)*(1 + r_z) - C[1:-1,i]
        B = diags(b_diag,offsets=0) +\
            diags(-r_z/2*np.ones(dim_z-3),offsets=1) +\
            diags(-r_z/2*np.ones(dim_z-3),offsets=-1)
        wave[1:-1] = splinalg.spsolve(B,d)
        wave_history[1:-1,i] = wave[1:-1]