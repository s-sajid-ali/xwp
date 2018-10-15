'''

This file is part of the zone_plate_testing repository and an exntension of the recipie to simulate one tilted zone plate fonund in the notebook simlate_zp_with_tilt.ipynb to a workflow which does the aforementioned task multiple times. 

'''

import numpy as np
import os,pickle,time
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float
from multislice import prop,prop_utils
from os.path import dirname as up


'''

make_zp_from_rings : make a zone plate from the rings which were created earlier.

Inputs : n - number of rings, grid_size 

Outputs : a numpy array containing the zone plate

'''
def make_zp_from_rings(n,grid_size):
    zp = np.zeros((grid_size,grid_size))
    for i in range(n):
        if i%2 == 1 :
            locs_ = np.load('ring_locs_'+str(i)+'.npy')
            locs_ = tuple((locs_[0],locs_[1]))
            vals_ = np.load('ring_vals_'+str(i)+'.npy')
            zp[locs_] = vals_
    return zp


'''

tilt : get the focal spot for a given zone plate and tilt angle of the input wave and save it to a tiff file

Inputs : i-tilt angle in degrees,zp - zone plate pattern, thickness (of the zone plate),parameters, out_dir - output directory

Outputs : dictionary containing :  tilt angle, max_loc - location of the zone plate, L2 - side length at output plane

'''

def tilt(i,zp,thickness,parameters,out_dir):
    zp_thickness = thickness 
    beta  = parameters['beta']
    delta = parameters['delta']    
    zp_coords = parameters['zp_coords']    
    step_xy = parameters['step_xy']
    energy = parameters['energy(in eV)']
    wavel = parameters['wavelength in m']
    f = parameters['focal_length']
    L = step_xy*np.shape(zp)[0] 
    n = np.shape(zp)[0]   
    
    os.chdir(out_dir)

    t1 = time.time()
    print('claclulating for tilt angle : ',i)      
    theta = (i)*(np.pi/180)
    slope = np.tan(theta)
    x = np.linspace(zp_coords[0],zp_coords[1],n)
    X,Y = np.meshgrid(x,x)
    z1 = 2*np.pi*(1/wavel)*slope*X
    wave_in = np.multiply(np.ones((n,n),dtype='complex64'),np.exp(1j*(z1)))
    print('step_xy :',step_xy,'wavelength :',wavel,'zp_thickness :',zp_thickness)
    number_of_steps_zp =  prop_utils.number_of_steps(step_xy,wavel,zp_thickness)*2
    wave_focus,L2 = prop_utils.optic_illumination(wave_in,zp,delta,beta,zp_thickness,step_xy,wavel,number_of_steps_zp,0,f)
    focal_spot,x_,y_,max_val = prop_utils.get_focal_spot(np.abs(wave_focus),grid_size)
    io.imsave('tilt_'+str(i)+'.tiff',img_as_float(focal_spot))
    t2 = time.time()
    
    print('tilt image number :',i,'time taken :',(t2-t1))
    
    return {'tilt_angle':i,'max_loc':np.array([x_,y_]),'L2':L2}


os.chdir(os.getcwd()+str('/rings'))
parameters = pickle.load(open('parameters.pickle','rb'))
grid_size = parameters['grid_size']

'''
 Creating the set of parameters for the simulation data set. Making a common output directory for the results.
'''

num_zones = np.array([250])
thickness = np.array([2e-6])
inputs = np.linspace(0,1,50)

output_dir = up(os.getcwd())+str('/output/')
os.mkdir(output_dir)

'''
Nested for loop to run the simluation over the variable number of zones, thicknesses and tilt angles. The output for each set of parameters is saved into a different folder.
'''

for var1 in range(len(num_zones)):
    n = num_zones[var1]
    zp =  make_zp_from_rings(n,grid_size)
    for var2 in range(len(thickness)) :
        print(num_zones[var1],' zones, ',thickness[var2] * 1e6,' microns thick')
        max_loc = []
        t = thickness[var2]
        out_dir = up(os.getcwd())+str('/output/')+ str('zones_'+str(n)+'_thickness_'+str(t))
        os.mkdir(out_dir)
        print(n,t,grid_size)
        for angle in inputs:
            max_loc.append(tilt(angle,zp,t,parameters,out_dir))
        np.save('max_loc.npy', max_loc) 
