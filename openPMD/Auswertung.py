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

class Auswertung:
    def __init__(self,filename,n_bins,bin_min,bin_max,species,slice_size,m,timestep):
        self.n_bins=n_bins
        self.bin_min=bin_min
        self.bin_max=bin_max
        self.filename=filename
        self.species=species
        self.slice=slice_size
        self.m=m
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        self.N_all=np.shape(f['/data/{}/particles/{}/momentum/x/'.format(timestep,species)])[0]
        f.close()
    
    def N(self,timestep):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        x = f['/data/{}/particles/{}/position/x'.format(timestep,self.species)][...][self.m] 
        f.close()
        return np.shape(x)[0]
    
    def Phasenraum(self,timestep,coordinate):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        handler  = f['/data/{}/particles/{}/'.format(timestep,self.species)]
        handlerE=f['/data/{}/fields/E'.format(timestep)] 
        positionSI=handler['position/'+coordinate].attrs['unitSI']/1e-6
        grit_size=np.array(np.shape(handlerE[coordinate]))
        gritSpacing=handlerE.attrs['gridSpacing']*handlerE.attrs['gridUnitSI']
        if coordinate == 'x': half=grit_size[2]/2*gritSpacing[2]
        if coordinate == 'y': half=0#grit_size[1]/2*gritSpacing[1]
        if coordinate == 'z': half=grit_size[0]/2*gritSpacing[0]
        momentum = (handler['momentum/'+coordinate][...][self.m])
        position = (handler['position/'+coordinate][...][self.m]+
                    handler['positionOffset/'+coordinate][...][self.m])*positionSI -half*1e6
        f.close()
        return(position,momentum)
    
   ####ChunkSize = self.slice --> kann eigentlich raus und dann alles umbenannt werden und chunksize, damit die namen nicht so doppelt sind...     
    def chunks(self,ChunkSize):
        N_Chunks=np.int(np.round(self.N_all/ChunkSize))
        startsUnfiltered=np.arange(0,N_Chunks+1)*ChunkSize
        startsFiltered=np.zeros(N_Chunks+1,dtype=np.int)
        for i in range(N_Chunks+1):
            startsUnfiltered_i=startsUnfiltered[i]
            startsFiltered[i]=np.int(np.sum(self.m[0:startsUnfiltered_i]))
        #print('startsUnfiltered=',startsUnfiltered,'startsFiltered = ',startsFiltered)
        return(startsUnfiltered,startsFiltered)

    def EnergyHistogram(self,timestep):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        handler  = f['/data/{}/particles/{}/'.format(timestep,self.species)] 
        w = f['/data/{}/particles/{}/weighting'.format(timestep,self.species)] 
        E = np.zeros(self.N(timestep))
        weights = np.zeros(self.N(timestep)) 
        convert = (handler['momentum/x'].attrs['unitSI'])/const.speed_of_light/const.electron_mass
        startsUnfiltered,startsFiltered=self.chunks(self.slice)
        for i in range(len(startsFiltered)-1):
            weights[startsFiltered[i]:startsFiltered[i+1]] = w[startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]
            pxsq = (handler['momentum/x'][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]/weights[startsFiltered[i]:startsFiltered[i+1]])**2 
            pysq = (handler['momentum/y'][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]/weights[startsFiltered[i]:startsFiltered[i+1]])**2
            pzsq = (handler['momentum/z'][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]/weights[startsFiltered[i]:startsFiltered[i+1]])**2
            usq = (pxsq+pysq+pzsq)*(convert**2)
            gamma = np.sqrt(1+usq)
            E[startsFiltered[i]:startsFiltered[i+1]]=(gamma-1)*0.511   
        f.close()   
        return((np.histogram(E, bins=self.n_bins, range=(self.bin_min,self.bin_max), weights=weights))[0])
    
    def calcEmittance(self,coordinate,timestep):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        N_all=np.shape(f['/data/{}/particles/{}/momentum/x/'.format(timestep,self.species)])[0]
        handler  = f['/data/{}/particles/{}/'.format(timestep,self.species)]
        w = handler['weighting']
        slices = np.arange(0, N_all, self.slice)
        uxSI=(handler['momentum/'+coordinate].attrs['unitSI'])/const.speed_of_light/const.electron_mass
        positionSI=handler['position/'+coordinate].attrs['unitSI']
        position_sq_s=0.
        pos_mom_s =0.
        momentum_sq_s = 0.
        startsUnfiltered,startsFiltered=self.chunks(self.slice)
        for i in range(len(startsFiltered)-1):
            w_s=w[startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]
            position=(handler['position/'+coordinate][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]]+ handler['positionOffset/'+coordinate][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]])*positionSI
            momentum=(handler['momentum/'+coordinate][startsUnfiltered[i]:startsUnfiltered[i+1]][self.m[startsUnfiltered[i]:startsUnfiltered[i+1]]])/w_s*uxSI          
            N_e_slice = np.size(w_s)
            if np.sum(w_s)>0:
                position_sq_s += N_e_slice * np.average((position**2), weights=w_s)
                pos_mom_s += N_e_slice * np.average((position*momentum), weights=w_s) 
                momentum_sq_s += N_e_slice * np.average((momentum**2), weights=w_s)
 
        No=self.N(timestep)
        if No>0:
            position_sq = position_sq_s /No
            pos_mom =  pos_mom_s /No
            momentum_sq = momentum_sq_s /No
            if position_sq*momentum_sq>pos_mom**2: 
                emit=np.sqrt(position_sq*momentum_sq-pos_mom**2)/1e-6
            else: emit=0
        else: 
            print('alle Teilchen wurden weggefiltert',timestep)
            emit=0
        f.close() 
        return(emit)

    def EField(self,timestep):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        handler = f['/data/{}/fields/E'.format(timestep)]
        grit_size=np.array(np.shape(handler['y']))
        convert = handler['y'].attrs['unitSI']
        E_y = handler['y'][grit_size[2]/2,:,:]
        gritSpacing=handler.attrs['gridSpacing']*handler.attrs['gridUnitSI']
        
        f.close()
        return (np.abs(E_y[:, :]*convert),gritSpacing,grit_size)
    
    def BField(self,timestep):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        handler = f['/data/{}/fields/B'.format(timestep)]
        grit_size=np.array(np.shape(handler['y']))
        convert = handler['y'].attrs['unitSI']
        B_y = handler['y'][grit_size[2]/2,:,:]
        gritSpacing=handler.attrs['gridSpacing']*handler.attrs['gridUnitSI']
        
        f.close()
        return (np.abs(B_y[:, :]*convert),gritSpacing,grit_size)
    
    def Density(self,timestep):
        f = h5py.File(self.filename+"/h5/h5_all_{}.h5".format(timestep), "r")
        handler = f['/data/{}/fields/e_chargeDensity'.format(timestep)]
        density = handler[...]
        convert = handler.attrs['unitSI']
        grit_size=np.array(np.shape(handler))
        gritSpacing=handler.attrs['gridSpacing']*handler.attrs['gridUnitSI']
        D = np.abs(density[grit_size[0]//2, :, :]*convert/1.6e-19)      
        f.close()
        return (D,gritSpacing,grit_size)

