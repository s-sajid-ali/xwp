#Finite Difference based wave propagation in 1D

import numpy as np
from scipy.sparse import diags
import scipy.sparse.linalg as splinalg
from tqdm import trange

all = ['finite_diff_1d_free_space',
      'finite_diff_1d_matter']

'''
Convention used here is the same as used in Christian Fuhse's thesis.
Wave propagates along the x-axis.

Fuhse thesis : https://ediss.uni-goettingen.de/bitstream/
               handle/11858/00-1735-0000-0006-B592-3/
               fuhse.pdf?sequence=1%20fuhse%20thesis
'''

'''
finite_diff_1d_free_space - Finite difference based 1D wave 
                            propagation in free space
                            
dim_x        : number of planes to propagate 
dim_z        : size of the 1d wave
r_z          : as per 3.15 in Fuhse thesis
wave         : value of the wavefield at current iteration
wave_history : value of the wavefield at all iterations
'''

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

'''
finite_diff_1d_matter - Finite difference based 1D wave 
                        propagation in matter
                            
dim_x        : number of planes to propagate 
dim_z        : size of the 1d wave
r_z          : as per 3.15 in Fuhse thesis
C            : as per 3.15 in Fuhse thesis
wave         : value of the wavefield at current iteration
wave_history : value of the wavefield at all iterations
'''
        
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