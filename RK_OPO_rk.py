import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
from __future__ import division
import ConfigParser

def read_parameters(filename):
    config = ConfigParser.RawConfigParser()
    try:
        config.readfp(open(filename))
    except IOError:
        raise IOError('cannot find INPUT.ini, exiting')
    return config

def read_input():
    param = read_parameters("INPUT.ini")

    kappa_C   = param.getfloat("params", "kappa_C")
    kappa_X   = param.getfloat("params", "kappa_X")
    delta     = param.getfloat("params", "delta")
    k_p       = param.getfloat("params", "k_p")
    omega_p   = param.getfloat("params", "omega_p")
    sigma_p   = param.getfloat("params", "sigma_p")
    f_p       = param.getfloat("params", "f_p")
    Lx        = param.getfloat("params", "Lx")
    Ly        = param.getfloat("params", "Ly")
    Nx        = param.getfloat("params", "Nx")
    Ny        = param.getfloat("params", "Ny")
    Nt        = param.getfloat("params", "Nt")
    tot_h     = param.getfloat("params", "tot_h")
    dxsav_rk  = param.getfloat("params", "dxsav_rk")
    eps_r     = param.getfloat("params", "eps_r")
    def_x_pos = param.getfloat("params", "def_x_pos")
    def_y_pos = param.getfloat("params", "def_y_pos")
    gv        = param.getfloat("params", "gv")

    ax = 2 * Lx / Nx
    ay = 2 * Ly / Ny
    norm = ax * ay
    norm_c = np.sqrt(Nx*Ny)/256/np.sqrt(Lx*Ly)*70
    f_p = f_p / norm_c
    return

def init_pot_c():
    #for old coordinates of (Nx/2,Ny/2) set def_x_pos = -ax, def_y_pos=-ay
    ix = int((def_x_pos + Lx) / ax + 1)
    iy = int((def_y_pos + Ly) / ay + 1)

    pot_c = 0
    pot_c(ix,iy)=gv
    return

def init_pump_th():
    #top hat pump

    do iy=1, Ny
       sy=-Ly+(iy-1)*ay
       do ix=1, Nx
          sx=-Lx+(ix-1)*ax
          pump_spatial(ix,iy)=f_p*0.5*&
               ( tanh((1.0_dp/10)*( sqrt(sx**2+sy**2)+sigma_p ))-&
               tanh((1.0_dp/10)*( sqrt(sx**2+sy**2)-sigma_p )) ) + zero
       end do
    end do

    do iy=1, Ny
       #sy=-Ly+(iy-1)*ay
       do ix=1, Nx
          sx=-Lx+(ix-1)*ax
          pump_spatial(ix,iy)= pump_spatial(ix,iy)*cos(k_p*sx)+I*pump_spatial(ix,iy)*sin(k_p*sx)
       end do
    end do
    return

def setg():
    DO j=1,(Ny/2+1)
       DO k=1,(Nx/2+1)
          kinetic(k,j)=pi**2*(&
               &(k-1)**2/(Lx**2)+(j-1)**2/(Ly**2))
       END DO
    END DO
    DO j=(Ny/2+2),Ny
       DO k=(Nx/2+2),Nx
          kinetic(k,j)=pi**2*( &
               & (k-1-Nx)**2/(Lx**2)+(j-1-Ny)**2/(Ly**2))
       END DO
    END DO
    DO j=1,(Ny/2+1)
       DO k=(Nx/2+2),Nx
          kinetic(k,j)=pi**2*(&
               &(k-1-Nx)**2/(Lx**2)+(j-1)**2/(Ly**2))
       END DO
    END DO
    DO j=(Ny/2+2),Ny
       DO k=1,(Nx/2+1)
          kinetic(k,j)=pi**2*(&
               &(k-1)**2/(Lx**2)+(j-1-Ny)**2/(Ly**2))
       END DO
    END DO
    return


def main():

    read_input()
    init_pot_c()

    pdb=0

    init_pump_th()
    init_pump_th()

    setg()


    x1_r=0
    x2_r=tot_h
    h1_r=0.001_dp
    hmin_r=0

    odeint_rk(pdb,x1_r,x2_r,eps_r,h1_r,hmin_r)
