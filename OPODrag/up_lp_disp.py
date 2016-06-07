# Dispersion: UP, LP, exciton, photon

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

omega_X = 1.4*1e+03
omega_C0 = 1.4003*1e+03
#delta = omega_C0 - omega_X

kz = 20.0

Omega_R = 0.0025*1e+03


nkx = 1024
nky = 1024
delta_k = 0.05

side_k = nkx * delta_k / 2
kx = ky = np.arange(-side_k, side_k, delta_k)
KX, KY = np.meshgrid(kx, ky)


mpl.rcParams.update({'font.size': 18,
                     'axes.titlesize': 20,
                     'font.family': 'serif',
                     'axes.labelsize': 22,
                     'legend.fontsize': 14,
                     'axes.linewidth': 1.5,
                     'font.serif': 'Computer Modern Roman',
                     'xtick.labelsize': 20,
                     'ytick.labelsize': 20,
                     'xtick.major.size': 5.5,
                     'ytick.major.size': 5.5,
                     'xtick.major.width': 1.5,
                     'ytick.major.width': 1.5,
                     'text.usetex': True})

def index_mom(mom):
    return int(np.floor((mom + side_k) / delta_k))

def enC(kx, ky):
    return (omega_C0 * np.sqrt(1 + (np.sqrt(kx**2 + ky**2) / kz)**2) - omega_X
            )

def enLP(kx, ky):
    return 0.5 * enC(
        kx, ky) - 0.5 * np.sqrt(enC(kx, ky)**2 + 4 * Omega_R**2)

def enUP(kx, ky):
    return 0.5 * enC(
        kx, ky) + 0.5 * np.sqrt(enC(kx, ky)**2 + 4 * Omega_R**2)


kxL, kxR = index_mom(0), index_mom(3)
C_band = enC(KX, KY)
LP = enLP(KX, KY)
UP = enUP(KX, KY)



fig_lp, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax2 = ax1.twiny()

ax1.plot(KX[nky / 2, kxL:kxR], C_band[nky / 2, kxL:kxR], color='r', ls = 'dashed', lw = 2)
ax1.plot(KX[nky / 2, kxL:kxR], LP[nky / 2, kxL:kxR], 'k', lw=2)
ax1.plot(KX[nky / 2, kxL:kxR], UP[nky / 2, kxL:kxR], 'k', lw=2)
ax1.axhline(y=0, color='r', ls='dashed', lw = 2) # exciton dispersion

plt.annotate(s='', xy=(0.1,UP[nky/2,kxL]), xytext=(0.1,LP[nky/2,kxL]), arrowprops=dict(arrowstyle='<->'))
ax1.text(0.15,0,r'$\delta$', fontsize=18)
ax1.text(0.05,0.5,r'$\sqrt{4\Omega_R^2 + \delta^2}$', transform=ax1.transAxes, fontsize=18)
ax1.text(0.5,0.1,'LP', transform=ax1.transAxes, fontsize=18)
ax1.text(0.3,0.8,'UP', transform=ax1.transAxes, fontsize=18)
ax1.text(0.5,0.35,'Exciton', transform=ax1.transAxes, fontsize=18)
ax1.text(0.5,0.7,'Photon', transform=ax1.transAxes, fontsize=18)

ax1.set_ylim(-3,6)
ax1.set_xlim(kx[kxL], kx[kxR])
ax1.set_xlabel(r'$k_x\,[\mu m^{-1}]$')

ax2Ticks = ax1.get_xticks()

hbarc = 0.19732697*1e+03 #meV*um 
def tick_function(ks):
    return ["%.1f" % np.degrees(np.arcsin(np.divide(hbarc*k,
                             LP[nky/2,index_mom(k)] + omega_X))) for k in ks]

ax2.set_xticks(ax2Ticks)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(tick_function(ax2Ticks))
ax2.set_xlabel(r'$\theta$ [deg]')

ax1.set_ylabel(r'$\omega-\omega_X(0)$ [meV]')
fig_lp.savefig('fig_up_lp.pdf', bbox_inches='tight', pad_inches=0.0, transparent=True)
