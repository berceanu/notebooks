import numpy as np
import scipy.constants as const
from ipywidgets import widgets
from IPython.display import display, clear_output
from ipywidgets import Button, Layout

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from opmd_viewer import OpenPMDTimeSeries
from opmd_viewer.addons import LpaDiagnostics

from multiprocessing import Pool

import h5py
#import h5filter
import Auswertung


class Emittance:
    def __init__(self,filename,species,slice_size,coordinate,m):
        self.filename=filename
        self.species=species
        self.slice=slice_size
        self.coordinate=coordinate
        self.m=m

    def calcEmittance(self,timestep):
        f = h5py.File(self.filename+"/h5/simData_{}.h5".format(timestep), "r")
        N_all=np.shape(f['/data/{}/particles/{}/momentum/x/'.format(timestep,self.species)])[0]
        handler  = f['/data/{}/particles/{}/'.format(timestep,self.species)]
        w = handler['weighting']
        slices = np.arange(0, N_all, self.slice)
        uxSI=(handler['momentum/'+self.coordinate].attrs['unitSI'])/const.speed_of_light/const.electron_mass
        positionSI=handler['position/'+self.coordinate].attrs['unitSI']
        position_sq_s=0.
        pos_mom_s =0.
        momentum_sq_s = 0.
#### Hier noch die Bin Größen raus nehmen        
        Calc=Auswertung.Auswertung(self.filename,1024,20,100,self.species,self.slice,self.m,timestep)
        startsUnfiltered,startsFiltered=Calc.chunks(self.slice)
        for i in range(len(startsFiltered)-1):
            w_s=w[startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]
            position=(handler['position/'+self.coordinate][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]+ handler['positionOffset/'+self.coordinate][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]])*positionSI
            momentum=(handler['momentum/'+self.coordinate][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]])/w_s*uxSI          
            N_e_slice = np.size(w_s)
            if np.sum(w_s)>0:
                position_sq_s += N_e_slice * np.average((position**2), weights=w_s)
                pos_mom_s += N_e_slice * np.average((position*momentum), weights=w_s) 
                momentum_sq_s += N_e_slice * np.average((momentum**2), weights=w_s)
 
        No=Calc.N(timestep)
        if No>0:
            position_sq = position_sq_s /No
            pos_mom =  pos_mom_s /No
            momentum_sq = momentum_sq_s /No
            emit=np.sqrt(position_sq*momentum_sq-pos_mom**2)/1e-6
        else: 
            print('alle Teilchen wurden weggefiltert')
            emit=0

        f.close() 
        return(emit)
    
