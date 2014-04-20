from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ConfigParser


def find_index(dist_or_mom, side, delta):
    return int(np.around((side + dist_or_mom) / delta))


def read_input(filename):
    param = ConfigParser.RawConfigParser()
    try:
        param.readfp(open(filename))
    except IOError:
        raise IOError('cannot find input file, exiting')

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

    norm_c = np.sqrt(Nx*Ny)/256/np.sqrt(Lx*Ly)*70
    f_p = f_p / norm_c

    return (kappa_C, kappa_X, delta, k_p, omega_p,
		    sigma_p, f_p, Lx, Ly, Nx, Ny, Nt,
		    tot_h, dxsav_rk, eps_r, def_x_pos, def_y_pos, gv)


def init_pot_c(iy, ix, strength, ny, nx):
    potential = np.zeros((ny, nx))
    potential[iy, ix] = strength
    return potential


def init_pump_th(fp, sigmap, kp, Y, X):
    #top hat pump
    r = np.sqrt(X**2 + Y**2)
    return 0.5 * fp * \
	    (np.tanh(1 / 10 * (r + sigmap)) - np.tanh(1 / 10 * (r - sigmap)))\
	    * np.exp(1j * kp * X)


def setg(KY, KX):
    return KY**2 + KX**2


#def main():
kappa_C, kappa_X, delta, k_p, omega_p,\
sigma_p, f_p, Lx, Ly, Nx, Ny, Nt,\
tot_h, dxsav_rk, eps_r, def_x_pos, def_y_pos, gv = read_input('INPUT.ini')

y, ay = np.linspace(-Ly, Ly, num=Ny, endpoint=False, retstep=True)
x, ax = np.linspace(-Lx, Lx, num=Nx, endpoint=False, retstep=True)
#norm = ax * ay
X, Y = np.meshgrid(x, y)

side_ky = np.pi / ay
side_kx = np.pi / ax
ky, delta_ky = np.linspace(-(2 * (Ny - 1) / float(Ny) - 1) * side_ky, side_ky, num=Ny, retstep=True)
kx, delta_kx = np.linspace(-(2 * (Nx - 1) / float(Nx) - 1) * side_kx, side_kx, num=Nx, retstep=True)
KX, KY = np.meshgrid(kx, ky)

idx_def_y_pos = find_index(def_y_pos, Ly, ay)
idx_def_x_pos = find_index(def_x_pos, Lx, ax)
pot_c = init_pot_c(idx_def_y_pos, idx_def_x_pos, gv, Ny, Nx)

pdb = np.zeros((Ny, Nx, Nt))

pump_spatial = init_pump_th(f_p, sigma_p, k_p, Y, X)
kinetic = setg(KY, KX)

x1_r=0
x2_r=tot_h
h1_r=0.001
hmin_r=0

#odeint_rk(pdb,x1_r,x2_r,eps_r,h1_r,hmin_r)
print 'done'


#if __name__ == "__main__":
    #main()
