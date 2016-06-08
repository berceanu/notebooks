# Optical limiter and bistable regimes
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

omega_X = 1.4
omega_C0 = 1.4
kz = 20.0
Omega_R = 0.0025
gp = 12e-05

k_pmpx = 1.6
k_pmpy = 0.0

def enC(kx, ky):
    return (omega_C0 * np.sqrt(1 + (np.sqrt(kx**2 + ky**2) / kz)**2) - omega_X
            ) / gp

def enLP(kx, ky):
    return 0.5 * enC(
        kx, ky) - 0.5 * np.sqrt(enC(kx, ky)**2 + 4 * Omega_R**2 / gp**2)

def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + ((Omega_R / gp) / (enLP(kx, ky) - enC(kx, ky)))**2)

ep = enLP(k_pmpx, k_pmpy)
xp = hopf_x(k_pmpx, k_pmpy)


# optical limiter
omega_pmp = 1.39875
omega_p_chosen = (omega_pmp - omega_X) / gp
print(omega_p_chosen*gp*1e+03)
#print omega_p_chosen - ep - np.sqrt(3)/2 #check

n_lim = np.arange(0,1.6,0.01)
ip_lim = [(((ep - omega_p_chosen + xp**2 * n)**2 + 1./4.) * n)/xp**4 for n in n_lim]


# optical bistability
omega_pmp = 1.39922
omega_p_chosen = (omega_pmp - omega_X) / gp
print(omega_p_chosen*gp*1e+03)
#print omega_p_chosen - ep - np.sqrt(3)/2 #check

n_bis = np.arange(0,5,0.01)
ip_bis = [(((ep - omega_p_chosen + xp**2 * n)**2 + 1./4.) * n)/xp**4 for n in n_bis]

mpl.rcParams.update({'font.size': 18,
                     'axes.titlesize': 20,
                     'font.family': 'serif',
                     'axes.labelsize': 30,
                     'legend.fontsize': 14,
                     'axes.linewidth': 3,
                     'font.serif': 'Computer Modern Roman',
                     'xtick.labelsize': 25,
                     'ytick.labelsize': 25,
                     'xtick.major.size': 11,
                     'ytick.major.size': 11,
                     'xtick.major.width': 3,
                     'ytick.major.width': 3,
                     'text.usetex': True})

fig_bistable, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].plot(ip_lim, n_lim, 'k', lw=3, zorder=1)

axes[1].plot(ip_bis[0:114], n_bis[0:114], 'k', lw=3, zorder=1)
axes[1].plot(ip_bis[115:330], n_bis[115:330], 'k', ls="dotted", lw=3, zorder=1)
axes[1].plot(ip_bis[331:-1], n_bis[331:-1], 'k', lw=3, zorder=1)


#axes[1].scatter(1.2,3.3, s=100, c='k', zorder=2)
axes[1].scatter(6.1,1.15, s=100, c='k', zorder=2)

axes[1].annotate(s='', xy=(1.2,0.1), xytext=(1.2,3.3),
                 arrowprops=dict(facecolor='red', width=2))
axes[1].annotate(s='', xy=(6.1,4.4), xytext=(6.1,1.15),
                 arrowprops=dict(facecolor='red', width=2))

axes[1].text(1.3,0.4,r'$\Delta_p > 0$', fontsize=30)
axes[1].text(6.5,1.1,'pump-only bistability point', fontsize=24)
axes[1].text(1.4,3.1,r'$\Delta_p = 0$', fontsize=30)
axes[1].text(4.5,4.5,r'$\Delta_p < 0$', fontsize=30)

axes[0].set_xlim(ip_lim[0], ip_lim[-1])
axes[1].set_xlim(ip_bis[0], ip_bis[-1])
axes[1].set_ylim(n_bis[0], n_bis[-1])


for ax in axes:
    ax.set_xlabel(r'$I_p\,[\gamma_p^3]$')
    ax.xaxis.set_ticks([0, 4, 8, 12, 16])

axes[0].set_ylabel(r'$\epsilon_p\,[\gamma_p]$')
axes[0].yaxis.set_ticks([0.4, 0.8, 1.2, 1.6])

fig_bistable.savefig('fig_bistable.pdf', bbox_inches='tight', pad_inches=0.0, transparent=True)
