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

import h5filter
import Auswertung
#import Emittance

class PlotFkt:
    def __init__(self, filename,species,slice_size,filterselection):
        self.filename=filename
        self.species=species
        self.slice_size=slice_size
        self.ts_2d = LpaDiagnostics(filename+'h5/')
    
    def get_globalPosition(self,timestep):
        f = h5py.File(self.filename+"h5/h5_all_{}.h5".format(timestep), "r")
        handler = f['/data/{}/fields/E'.format(timestep)]
        gP=np.round((handler.attrs['gridGlobalOffset']*handler.attrs['gridUnitSI']*1e6)[1],decimals=2)
        #print('bei',np.round((handler.attrs['gridGlobalOffset']*handler.attrs['gridUnitSI']*1e6)[1],decimals=2),'µm')
        f.close()  
        return(gP)

    def createCheckbox(self,checkboxlist,text):
        containerList=[(widgets.Checkbox(description=k)) for k in checkboxlist]
        containerList.insert(0, widgets.HTML(value=text))
        bigContainer = widgets.Box(children=(containerList))
        display(bigContainer)
        return  bigContainer  

    def filtern(self,FilterDictValues,timestep):
        neuerFilter=h5filter.h5filter(self.filename,timestep,self.species,self.slice_size)
        if "gamma" in FilterDictValues.keys():
            remainingGamma=neuerFilter.filterGamma(FilterDictValues['gamma'][0],FilterDictValues['gamma'][1],self.species)
          #  print('nach dem Gamma Filter sind',remainingGamma,'% der Teilchen übrig')
        if "x" in FilterDictValues.keys(): 
            remainingX=neuerFilter.filterPosition('x',FilterDictValues['x'][0],FilterDictValues['x'][1],timestep)
           # print('nach dem x Filter sind',remainingX,'% der Teilchen übrig')
        if "y" in FilterDictValues.keys(): 
            remainingY=neuerFilter.filterPosition('y',FilterDictValues['y'][0],FilterDictValues['y'][1],timestep)
           # print('nach dem y Filter sind',remainingY,'% der Teilchen übrig')
        if "z" in FilterDictValues.keys(): 
            remainingZ=neuerFilter.filterPosition('z',FilterDictValues['z'][0],FilterDictValues['z'][1],timestep)
           # print('nach dem z Filter sind',remainingZ,'% der Teilchen übrig')
        if "ux" in FilterDictValues.keys(): 
            remainingUx=neuerFilter.filterMomentum('x',FilterDictValues['ux'][0],FilterDictValues['ux'][1])
           # print('nach dem ux Filter sind',remainingUx,'% der Teilchen übrig')
        if "uy" in FilterDictValues.keys(): 
            remainingUz=neuerFilter.filterMomentum('z',FilterDictValues['uz'][0],FilterDictValues['uz'][1])
           # print('nach dem uz Filter sind',remainingUz,'% der Teilchen übrig')
        return(neuerFilter.maske)

    def workEmittance(self,args):
        timestep, FilterDictValues = args
        m=self.filtern(FilterDictValues,timestep)
        b=Auswertung.Auswertung(self.filename,1024,20,100,self.species,self.slice_size,m,timestep)
        return(b.calcEmittance('x',timestep))

    def workEnergyHist(self,args):
        timestep, FilterDictValues = args
        m=self.filtern(FilterDictValues,timestep)
        b=Auswertung.Auswertung(self.filename,1024,0,150,self.species,self.slice_size,m,timestep)
        return(b.EnergyHistogram(timestep))
    
    def workSliceEmittance(self,args2):
        i, timestep, FilterDictValues,L,globalPosition = args2      
        y1=globalPosition+i*L/10
        if i<10: 
            y2=globalPosition+(i+1)*L/10
            neuerFilter.filterPosition('y',y1*1e-6,y2*1e-6,timestep)
            m=self.filtern(FilterDictValues,timestep)
        if i==10: 
            neuerFilter.filterPosition('y',y1*1e-6,(y1+L/2)*1e-6,timestep)
            m=self.filtern(FilterDictValues,timestep)
        y1 -= globalPosition 
        b=Auswertung.Auswertung(self.filename,1024,0,150,self.species,self.slice_size,m,timestep)
        emitS=b.calcEmittance('x',timestep)
        return(y1,emitS)
    
    def SliceEmittance(self,args):#timestep, FilterDictValues):
        timestep, FilterDictValues = args       
        f = h5py.File(self.filename+"/h5/simData_{}.h5".format(timestep), "r")
        handler  = f['/data/{}/particles/{}/'.format(timestep,self.species)]
        w = handler['weighting']   
        handlerE=f['/data/{}/fields/E'.format(timestep)] 
        grit_size=np.array(np.shape(handlerE['x']))
        gritSpacing=handlerE.attrs['gridSpacing']*handlerE.attrs['gridUnitSI']
        globalPosition=np.round((handlerE.attrs['gridGlobalOffset']*handlerE.attrs['gridUnitSI']*1e6)[1],decimals=2)
        f.close() 
        L=grit_size[1]*gritSpacing[1]*1e6
        y = np.zeros(11)
        emitS = np.zeros(11) 
        
        #args2 = []
        #for i in range(0,11,1):
        #    args.append([i,timestep, FilterDictValues,L,globalPosition])
        #pool = Pool(15)
        #work=pool.map(self.workSliceEmittance,args2)
        #for index in range(0,11,1):
        #    y[index],emitS[index] = work[index] 
                
        for i in range(0,11,1):
            neuerFilter=h5filter.h5filter(self.filename,timestep,self.species,self.slice_size)
            self.filtern(FilterDictValues,timestep)
            y[i]=globalPosition+i*L/10
            if i<10: 
                y[i+1]=globalPosition+(i+1)*L/10
                neuerFilter.filterPosition('y',y[i]*1e-6,y[i+1]*1e-6,timestep) 
                #m=self.filtern(FilterDictValues,timestep)
            if i==10: 
                neuerFilter.filterPosition('y',y[i]*1e-6,(y[i]+L/2)*1e-6,timestep)
               # m=self.filtern(FilterDictValues,timestep)
            y[i] -= globalPosition 
            m=neuerFilter.maske
            b=Auswertung.Auswertung(self.filename,1024,0,150,self.species,self.slice_size,m,timestep)
            emitS[i]=b.calcEmittance('x',timestep)   
        print('timestep=',timestep,'emitS=',emitS)
        return(y,emitS)
    
        