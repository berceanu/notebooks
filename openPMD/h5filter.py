import numpy as np
import scipy.constants as const
from ipywidgets import widgets
from IPython.display import display, clear_output
from ipywidgets import Button, Layout

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from opmd_viewer import OpenPMDTimeSeries
from opmd_viewer.addons import LpaDiagnostics

import h5py

class h5filter:
    def __init__(self,filename,timestep,species,slice_size):
        self.filename=filename
        self.timestep=timestep
        self.species=species
        self.slice=slice_size
        f = h5py.File(self.filename+"h5/h5_all_{}.h5".format(timestep), "r")
        self.N = np.shape(f['/data/{}/particles/{}/momentum/x/'.format(timestep,species)])[0]
        self.maske=np.ones(self.N, dtype=np.bool)
        self.slices = np.arange(0, self.N, self.slice)
        print('N vor Filter=',self.N)
        f.close()
        
    def filterPosition(self, coordinate, position_min, position_max, timestep):
        f = h5py.File(self.filename + "/h5/h5_all_{}.h5".format(timestep), "r")
        handlerE = f['/data/{}/fields/E'.format(timestep)]
        grid_size=np.array(np.shape(handlerE[coordinate]))
        gridSpacing=handlerE.attrs['gridSpacing']*handlerE.attrs['gridUnitSI']       
        if coordinate == 'x': half=grid_size[2]/2*gridSpacing[2]
        if coordinate == 'y': half=0#grid_size[1]/2*gridSpacing[1]
        if coordinate == 'z': half=grid_size[0]/2*gridSpacing[0]
        handler = f['/data/{}/particles/e/position/'.format(self.timestep)]
        handlero = f['/data/{}/particles/e/positionOffset/'.format(self.timestep)]
        for start_slice in self.slices:
            #print("slice starts at: {}".format(start_slice))
            position=(handler['{}'.format(coordinate)][start_slice:start_slice+self.slice]+
                      handlero['{}'.format(coordinate)][start_slice:start_slice+self.slice])* handler['{}'.format(coordinate)].attrs['unitSI']-half    
            self.maske[start_slice:start_slice+self.slice] *= np.less_equal(position, position_max)*np.greater_equal(position, position_min)        
        #print('nach dem ',coordinate,' Filter sind',np.shape(np.where(self.maske==True))[1]/self.N*100, '% der Teilchen übrig,  N = ',np.shape(np.where(self.maske==True))[1])
        f.close()
        return(np.shape(np.where(self.maske==True))[1]/self.N*100)
        
    def filterGamma(self,gamma_min=0,gamma_max=100,species='e'):
        f = h5py.File(self.filename + "/h5/h5_all_{}.h5".format(self.timestep), "r")     
        w = f['/data/{}/particles/{}/weighting/'.format(self.timestep,species)]
        handler = f['/data/{}/particles/e/momentum/'.format(self.timestep)]
        convert = (handler['y'].attrs['unitSI'])/const.speed_of_light/const.electron_mass
        for i, start_slice in enumerate(self.slices):
            #print("slice starts at: {}".format(start_slice))
            print('{} slices remaining..'.format(self.slices.size - i))
            psq = ((handler['x'][start_slice:start_slice+self.slice]/w[start_slice:start_slice+self.slice])**2 + 
                   (handler['y'][start_slice:start_slice+self.slice]/w[start_slice:start_slice+self.slice])**2 + 
                   (handler['z'][start_slice:start_slice+self.slice]/w[start_slice:start_slice+self.slice])**2)
            usq = psq*(convert**2)
            gamma = np.sqrt(1+usq)
            self.maske[start_slice:start_slice+self.slice] *= np.less_equal(gamma, gamma_max)*np.greater_equal(gamma, gamma_min)       
        #print('nach gamma Filter sind',np.shape(np.where(self.maske==True))[1]/self.N*100, '% der Teilchen übrig, N = ',np.shape(np.where(self.maske==True))[1])
        f.close()
        return(np.shape(np.where(self.maske==True))[1]/self.N*100)
    
    def filterMomentum(self, coordinate, u_min=-10, u_max=10):
        f = h5py.File(self.filename + "/h5/h5_all_{}.h5".format(self.timestep), "r")
        handleru = f['/data/{}/particles/e/momentum/{}'.format(self.timestep,coordinate)]       
        for start_slice in self.slices:
            #print("slice starts at: {}".format(start_slice))
            u=handleru[start_slice:start_slice+self.slice]
            self.maske[start_slice:start_slice+self.slice] *= np.less_equal(u, u_max)*np.greater_equal(u, u_min) 
        #print('nach u',coordinate,' Filter sind',np.shape(np.where(self.maske==True))[1]/self.N *100, '% der Teilchen übrig, N = ',np.shape(np.where(self.maske==True))[1])
        f.close() 
        return(np.shape(np.where(self.maske==True))[1]/self.N)
    
   # def readposition(self,coordinate):
   #     f = h5py.File(self.filename + "/h5/h5_all_{}.h5".format(self.timestep), "r")
   #     handler = f['/data/{}/particles/{}'.format(self.timestep, self.species)]
   #     convert=handler['position/{}'.format(coordinate)].attrs['unitSI']
   #     position=(handler['position/{}'.format(coordinate)][...][self.maske]+
   #               handler['positionOffset/{}'.format(coordinate)][...][self.maske])* convert
   #     f.close()
   #     #print('ausgangswert:',np.shape(np.where(self.maske==True))[1]/self.N)
   #     return position
   #
   # def Number(self):
   #     f = h5py.File(self.filename + "/h5/h5_all_{}.h5".format(self.timestep), "r")
   #     handler = f['/data/{}/particles/e'.format(self.timestep)]
   #     x=(handler['position/x'][...][self.maske]) 
   #     f.close()
   #     return (np.shape(x)[0])