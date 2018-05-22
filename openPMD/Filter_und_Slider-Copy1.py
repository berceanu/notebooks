import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from opmd_viewer.addons import LpaDiagnostics

from multiprocessing import Pool


import Auswertung
import Emittance
import PlotFkt

def createFilter(c):
    clear_output()
    display(tslider, filterCheckbox, button1)
    #gP=Darstellungen.get_globalPosition(timestep.value)
    style = {'description_width': 'initial'}
    gamma=widgets.VBox(children=(widgets.IntSlider(min=0,max=100,step=1,value=0,description='g_min ',style=style),
                                 widgets.IntSlider(min=20,max=250,step=1,value=250,description='g_max ',style=style)))
    x=widgets.VBox(children=(widgets.FloatSlider(min=-20,max=20,step=0.1,value=-5,description='x_min [µm]',style=style),
                             widgets.FloatSlider(min=-20,max=20,step=0.1,value=5,description='x_max [µm]',style=style)))
    y=widgets.VBox(children=(widgets.FloatSlider(min=0,max=400,step=0.1,value=-5,description='y_min [µm]',style=style),
                             widgets.FloatSlider(min=0,max=400,step=0.1,value=5,description='y_max [µm]',style=style)))
    z=widgets.VBox(children=(widgets.FloatSlider(min=-20,max=20,step=0.1,value=-5,description='z_min [µm]',style=style),
                             widgets.FloatSlider(min=-20,max=20,step=0.1,value=5,description='z_max [µm]',style=style)))
    ux=widgets.VBox(children=(widgets.FloatSlider(min=-10,max=10,step=0.1,value=-5,description='ux_min[gamma beta]',style=style),
                              widgets.FloatSlider(min=-10,max=10,step=0.1,value=5,description='ux_max[gamma beta]',style=style)))
    uz=widgets.VBox(children=(widgets.FloatSlider(min=-10,max=10,step=0.1,value=-5,description='uz_min[gamma beta]',style=style),
                              widgets.FloatSlider(min=-10,max=10,step=0.1,value=5,description='uz_max[gamma beta]',style=style)))
    FilterDict={}
    if filterCheckbox.children[1].value: FilterDict['gamma'] = gamma
    if filterCheckbox.children[2].value: FilterDict['x'] = x
    if filterCheckbox.children[3].value: FilterDict['y'] = y
    if filterCheckbox.children[4].value: FilterDict['z'] = z
    if filterCheckbox.children[5].value: FilterDict['ux'] = ux
    if filterCheckbox.children[6].value: FilterDict['uy'] = uz
    
    FilterContainer=widgets.Box(children=[FilterDict[name] for name in FilterDict.keys()])
    display(FilterContainer)
    
    Energyhistogramselection=['Energyhistogram je Zeitschritt','Energyhistogramm im Verlauf']
    EnergyContainer=Darstellungen.createCheckbox(Energyhistogramselection,'<b>Energyhistogramm:</b>')
    Phasenraumselection=['Physenraum ux-x','Physenraum uz-z','Physenraum uy-y']
    PhasenraumContainer=Darstellungen.createCheckbox(Phasenraumselection,'<b>Phasenräume:</b>')
    Emittanceselection=['Emittance x','Emittance y','Emittance z', 'Emittance Verlauf','Slice Emittance (x)']
    EmittanceContainer=Darstellungen.createCheckbox(Emittanceselection,'<b>Emittance:</b>') 
    Fieldselection=['E-Feld','B-Feld', 'Plasma-Dichte']
    FieldContainer=Darstellungen.createCheckbox(Fieldselection,'<b>Felder:</b>') 
  
    def useFilter(sophie):
        clear_output()
        display(tslider, filterCheckbox, button1,FilterContainer,EnergyContainer,PhasenraumContainer,EmittanceContainer,FieldContainer,button2)
        
        #### Filtern:
        FilterDictValues={}
        if filterCheckbox.children[1].value: 
            FilterDictValues['gamma'] = (gamma.children[0].value,gamma.children[1].value)
        if filterCheckbox.children[2].value: 
            FilterDictValues['x'] = [x.children[0].value*1e-6, x.children[1].value*1e-6]
        if filterCheckbox.children[3].value: 
            FilterDictValues['y'] = [y.children[0].value*1e-6, y.children[1].value*1e-6]
        if filterCheckbox.children[4].value: 
            FilterDictValues['z'] = [z.children[0].value*1e-6, z.children[1].value*1e-6]
        if filterCheckbox.children[5].value: 
            FilterDictValues['ux'] = [ux.children[0].value, ux.children[1].value]
        if filterCheckbox.children[6].value: 
            FilterDictValues['uz'] = [uz.children[0].value, uz.children[1].value]
        m=Darstellungen.filtern(FilterDictValues,timestep.value)
        b=Auswertung.Auswertung(filename,1024,20,150,species,slice_size,m,timestep.value)
        
        args = []
        for i in (ts_2d.iterations):
            args.append([i, FilterDictValues])
            
        ####Darstellungen:
        if EnergyContainer.children[1].value:
            bins = np.linspace(b.bin_min, b.bin_max, b.n_bins)
            plt.figure(num='Energyhistogram')
            plt.plot(bins,b.EnergyHistogram(timestep.value))
            plt.ylim((0,2e7))
            #plt.yscale('log')
            plt.xlabel(r'E$_\mathrm{kin}$ [MeV]')
            plt.ylabel(r'Anzahl Elektronen')
            plt.show()
        if EnergyContainer.children[2].value:
            pool = Pool(15)
            work=pool.map(Darstellungen.workEnergyHist,args)
            Energy_list = np.zeros((1024,len(ts_2d.iterations)))
            for index, iterT in enumerate(ts_2d.iterations):
                Energy_list[:,index] = work[index]
            plt.figure(num='Energyverlauf')
            timebins = np.linspace(0, 10000, 11)
            enegybins=np.linspace(0, 150, 1024)
            plt.pcolormesh(timebins, enegybins, Energy_list, norm=LogNorm(), vmin=1e6, vmax=1e8)
            plt.colorbar()
        if PhasenraumContainer.children[1].value or PhasenraumContainer.children[2].value or PhasenraumContainer.children[3].value:  
            plt.figure(figsize=(13.5,4.5),num='Phasenraum')
            if PhasenraumContainer.children[1].value:
                positionx, momentumx = b.Phasenraum(timestep.value,'x')
                plt.subplot(131) 
                plt.hist2d(positionx, momentumx , bins=256, norm=LogNorm())
                plt.xlabel(r'x [$\mu$m]')
                plt.ylabel(r'ux [$\beta \gamma$]')
                plt.colorbar()
            if PhasenraumContainer.children[2].value:
                positionz, momentumz = b.Phasenraum(timestep.value,'z')
                plt.subplot(132) #sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
                plt.hist2d(positionz, momentumz , bins=256, norm=LogNorm())
                plt.xlabel(r'z [$\mu$m]')
                plt.ylabel(r'uz [$\beta \gamma$]')
                plt.colorbar()
                plt.tight_layout(pad=0.4, w_pad=1.5)
            if PhasenraumContainer.children[3].value:
                positiony, momentumy = b.Phasenraum(timestep.value,'y')
                plt.subplot(133) 
                plt.hist2d(positiony, momentumy , bins=256, norm=LogNorm())
                plt.xlabel(r'y [$\mu$m]')
                plt.ylabel(r'uy [$\beta \gamma$]')
                plt.colorbar()
            plt.show()
        if EmittanceContainer.children[1].value:
            e=Emittance.Emittance(filename,species,slice_size,'x',m)
            print('Emittance x=',e.calcEmittance(timestep.value),'pi mm mrad')
        if EmittanceContainer.children[2].value:
            e=Emittance.Emittance(filename,species,slice_size,'y',m)
            print('Emittance y=',e.calcEmittance(timestep.value),'pi mm mrad')
        if EmittanceContainer.children[3].value:
            e=Emittance.Emittance(filename,species,slice_size,'z',m)
            print('Emittance z=',e.calcEmittance(timestep.value),'pi mm mrad')
        if EmittanceContainer.children[4].value: 
            emit_x_list = np.zeros(len(ts_2d.iterations))
            pool = Pool(15)
            work=pool.map(Darstellungen.workEmittance,args)
            for index, iterT in enumerate(ts_2d.iterations):
                emit_x_list[index] = work[index]           
            plt.figure(num='Emittance')
            plt.plot(ts_2d.iterations[1:] ,emit_x_list[1:])
            plt.ylabel(r'$\epsilon_{n,x}$ [$\pi$ mm mrad]')
            plt.xlabel('Zeitschritt')
            plt.show() 
        if EmittanceContainer.children[5].value:
            pool = Pool(15)
            work=pool.map(Darstellungen.SliceEmittance,args)           
            plt.figure(figsize=(10,17),num='normierte slice Emittance in pi mm mrad')#'Verlauf Slice Emittance')
            i=0
            l=len(work)
            ylim_j=np.zeros(l+1)
            for res in work:
                y_j,emitS_j =res
                print('i=',i,'emitS_j=',emitS_j)
                ylim_j[i]=np.max(emitS_j)
                i+=1
            ylim=np.max(ylim_j)
            i=0
            for res in work:
                y_i,emitS =res #Darstellungen.SliceEmittance(timestep.value, FilterDictValues)
                print('i=',i,'emitS=',emitS)
                plt.subplot(l,1,i+1)
                plt.plot(y_i,emitS)#,label=np.str(ts_2d.iterations[i]))
                plt.ylabel(np.str(ts_2d.iterations[i]))  #plt.ylabel(r'$\epsilon_{n,x}$ [$\pi$ mm mrad]')
                if ylim>0: plt.ylim(0,ylim)
                i+=1          
            plt.xlabel('y in µm')
            plt.show()
        if FieldContainer.children[1].value or FieldContainer.children[2].value or FieldContainer.children[3].value:  
            plt.figure(figsize=(13.5,4.5),num='Felder')
            if FieldContainer.children[1].value:
                E_y,gritSpacing,grit_size=b.EField(timestep.value)
                plt.subplot(131) 
                plt.title('E-Feld')
                plt.imshow(E_y, aspect="auto", origin="lower", extent=(-0.5*grit_size[0]*gritSpacing[0]*1e6, +0.5*grit_size[0]*gritSpacing[0]*1e6, 0, grit_size[1]*gritSpacing[1]*1e6))#, norm=LogNorm())
                plt.colorbar()
                plt.xlabel(r"$x \, \mathrm{[\mu m]}$")
                plt.ylabel(r"$y \, \mathrm{[\mu m]}$")
            if FieldContainer.children[2].value:
                B_y,gritSpacing,grit_size=b.BField(timestep.value)
                plt.subplot(132) 
                plt.title('B-Feld')
                plt.imshow(B_y, aspect="auto", origin="lower", extent=(-0.5*grit_size[0]*gritSpacing[0]*1e6, +0.5*grit_size[0]*gritSpacing[0]*1e6, 0, grit_size[1]*gritSpacing[1]*1e6))#, norm=LogNorm())
                plt.colorbar()
                plt.xlabel(r"$x \, \mathrm{[\mu m]}$")
                plt.ylabel(r"$y \, \mathrm{[\mu m]}$")
                plt.tight_layout(pad=0.4, w_pad=1.5)
            if FieldContainer.children[3].value:
                D,gritSpacing,grit_size=b.Density(timestep.value)
                plt.subplot(133) 
                plt.title('Charge-Density')
                plt.imshow(D, aspect="auto", origin="lower",norm=LogNorm(),extent=(-0.5*grit_size[0]*gritSpacing[0]*1e6, +0.5*grit_size[0]*gritSpacing[0]*1e6, 0, grit_size[1]*gritSpacing[1]*1e6))
                plt.colorbar()
                plt.xlabel(r"$x \, \mathrm{[\mu m]}$")
                plt.ylabel(r"$y \, \mathrm{[\mu m]}$")
            plt.show()
            
    button2 = widgets.Button(description = 'Anwenden')
    button2.on_click(useFilter)
    display(button2)
    
def on_value_change(change):
        clear_output()
        with out:
            clear_output()
            gP=Darstellungen.get_globalPosition(timestep.value)
            print('bei',gP,'µm')
        #tslider=widgets.HBox([timestep,out])
        display(tslider)
        filterCheckbox = Darstellungen.createCheckbox(filterselection,'<b>Filter:</b>')
        display(button1)


# In[6]:


filename="/media/bigdata_runs/initial/simOutput/"
#filename="/bigdata/hplsim/external/bercea20/runs/initial/simOutput/"

species='e'
#slice_size = 1024**2
slice_size = 1024
ts_2d = LpaDiagnostics(filename+'h5/')


class timestep:
    __slots__ = 'value'

timestep = timestep()
timestep.value = 5000


#FilterDictValues
FDV = {'gamma': (0, 250),
       # coords in meters
       'x': [-5*1e-6, 5*1e-6],
       'y': [-5*1e-6, 5*1e-6],
       'z': [-5*1e-6, 5*1e-6],
       # units gamma beta
       'ux': [-5, 5],
       'uz': [-5, 5]}


filterselection = ['Gamma', 'x','y','z', 'ux','uz']
Darstellungen=PlotFkt.PlotFkt(filename,species,slice_size,filterselection)

# line 70
m=Darstellungen.filtern(FDV,timestep.value)
b=Auswertung.Auswertung(filename,1024,20,150,species,slice_size,m,timestep.value)

filterselection = ['Gamma', 'x','y','z', 'ux','uz']



filterCheckbox = Darstellungen.createCheckbox(filterselection,'<b>Filter:</b>')

button1.on_click(createFilter)





