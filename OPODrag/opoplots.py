# Bogoliubov spectrum
# Mean-field: signal, pump, idler
# Mean-field: pump-only
# Dispersion: UP, LP, exciton, photon

import numpy as np
from scipy import optimize

import matplotlib as mpl
import matplotlib.pyplot as plt


from phcpy.phcpy2c import py2c_set_seed
from phcpy.solver import solve
from phcpy.solutions import strsol2dict

import ConfigParser
import sys

# reading parameters
def read_parameters(filename):
    config = ConfigParser.RawConfigParser()
    try:
        config.readfp(open(filename))
    except IOError:
        raise IOError('cannot find parameters.ini, exiting')
    return config

param = read_parameters("/home/berceanu/notebooks/OPODrag/parameters.ini")
param_set = sys.argv[1]

omega_X = 1.4
omega_C0 = 1.4
kz = param.getfloat(param_set, "kz")
Omega_R = param.getfloat(param_set, "Omega_R")
gamma_X = param.getfloat(param_set, "gamma_X")
gamma_C = param.getfloat(param_set, "gamma_C")
k_pmpx = param.getfloat(param_set, "k_pmpx")
k_pmpy = param.getfloat(param_set, "k_pmpy")
k_sigx = param.getfloat(param_set, "k_sigx")
k_sigy = param.getfloat(param_set, "k_sigy")
omega_pmp = param.getfloat(param_set, "omega_pmp")
ip_chosen = param.getfloat(param_set, "ip_chosen")
gv = param.getfloat(param_set, "gv")

nkx = 512
nky = 512
ipx_start = param.getfloat(param_set, "ipx_start")
ipx_end = param.getfloat(param_set, "ipx_end")
kxl = param.getfloat(param_set, "kxl")
kxr = param.getfloat(param_set, "kxr")
kyl = param.getfloat(param_set, "kyl")
kyr = param.getfloat(param_set, "kyr")
eigs_threshold = param.getfloat(param_set, "eigs_threshold")
stride_r = param.getint(param_set, "stride_r")
stride_c = param.getint(param_set, "stride_c")
delta_k = 0.005
ipstabini = param.getfloat(param_set, "ipstabini")
ipstabfin = param.getfloat(param_set, "ipstabfin")

k_idlx = 2 * k_pmpx - k_sigx
k_idly = 2 * k_pmpy - k_sigy

ks = ("{0:.3f}".format(k_sigx)).replace(".", "_")

gp = gamma_C + (1 / np.sqrt(1 + (Omega_R / ((0.5 * ((omega_C0 * np.sqrt(1 + (
    np.sqrt(k_pmpx**2 + k_pmpy**2) / kz)**2)) + omega_X) - 0.5 * np.sqrt(
        ((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx**2 + k_pmpy**2) / kz)**2)) -
         omega_X)**2 + 4 * Omega_R**2)) - (omega_C0 * np.sqrt(1 + (np.sqrt(
             k_pmpx**2 + k_pmpy**2) / kz)**2))))**2))**2 * (gamma_X - gamma_C)

omega_p_chosen = (omega_pmp - omega_X) / gp


side_k = nkx * delta_k / 2
kx = ky = np.arange(-side_k, side_k, delta_k)
print kx[-1]

KX, KY = np.meshgrid(kx, ky)

ipx = np.linspace(ipx_start, ipx_end, 30)


color_spi = ['blue', 'red', 'green']
marker_spi = ['^', 'o', 'v']
momentum_spi = np.array([[k_sigx, k_sigy], [k_pmpx, k_pmpy], [k_idlx, k_idly]])


def null(A, eps=1e-10):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def index_mom(mom):
    return int(np.floor((mom + side_k) / delta_k))


def enC(kx, ky):
    return (omega_C0 * np.sqrt(1 + (np.sqrt(kx**2 + ky**2) / kz)**2) - omega_X
            ) / gp


def enLP(kx, ky):
    return 0.5 * enC(
        kx, ky) - 0.5 * np.sqrt(enC(kx, ky)**2 + 4 * Omega_R**2 / gp**2)

def enUP(kx, ky):
    return 0.5 * enC(
        kx, ky) + 0.5 * np.sqrt(enC(kx, ky)**2 + 4 * Omega_R**2 / gp**2)


def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + ((Omega_R / gp) / (enLP(kx, ky) - enC(kx, ky)))**2)


def blue_enLP(kx, ky):
    return enLP(kx, ky) + 2 * hopf_x(kx, ky) ** 2 *\
        (ni_chosen + np_chosen + ns_chosen)

def gamma(kx, ky):
    return (gamma_C + hopf_x(kx, ky)**2 * (gamma_X - gamma_C)) / gp


es = enLP(k_sigx, k_sigy)
ep = enLP(k_pmpx, k_pmpy)
ei = enLP(k_idlx, k_idly)

gs = gamma(k_sigx, k_sigy)
gi = gamma(k_idlx, k_idly)

xs = hopf_x(k_sigx, k_sigy)
xp = hopf_x(k_pmpx, k_pmpy)
xi = hopf_x(k_idlx, k_idly)

alpha = (xi**2 / xs**2) * (gs / gi)


def n_hom_mf(ips):
    return np.array([optimize.brentq(
        lambda n: ((ep - omega_p_chosen + xp**2 * n)**2 + 1 / 4) * n - xp**4 * ip,
        0, 3) for ip in ips])


def L(kx, ky):
    return np.array([
        [-omega_s_chosen + enLP(k_sigx + kx, k_sigy + ky) - 1j * 1 / 2 * gamma(
            k_sigx + kx, k_sigy + ky) + 2 * (ni_chosen + np_chosen + ns_chosen)
         * hopf_x(k_sigx + kx, k_sigy + ky)**2, 2 * (
             p * np.conjugate(i) + s * np.conjugate(p)) * hopf_x(
                 k_pmpx + kx, k_pmpy + ky) * hopf_x(k_sigx + kx, k_sigy + ky),
         2 * s * np.conjugate(i) * hopf_x(k_idlx + kx, k_idly + ky) * hopf_x(
             k_sigx + kx, k_sigy + ky), s**2 * hopf_x(
                 k_sigx - kx, k_sigy - ky) * hopf_x(k_sigx + kx, k_sigy + ky),
         2 * p * s * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(
             k_sigx + kx, k_sigy + ky), (p**2 + 2 * i * s) * hopf_x(
                 k_idlx - kx, k_idly - ky) * hopf_x(k_sigx + kx, k_sigy + ky)],
        [2 * (i * np.conjugate(p) + p * np.conjugate(s)) *
         hopf_x(k_pmpx + kx, k_pmpy + ky) * hopf_x(k_sigx + kx, k_sigy + ky),
         -omega_p_chosen + enLP(k_pmpx + kx, k_pmpy + ky) - 1j * 1 / 2 * gamma(
             k_pmpx + kx, k_pmpy + ky) + 2 *
         (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_pmpx + kx, k_pmpy + ky)
         **2, 2 * (p * np.conjugate(i) + s * np.conjugate(p)) * hopf_x(
             k_idlx + kx, k_idly + ky) * hopf_x(k_pmpx + kx,
                                                k_pmpy + ky), 2 * p * s *
         hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), (
             p**2 + 2 * i * s) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(
                 k_pmpx + kx, k_pmpy + ky), 2 * i * p * hopf_x(
                     k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx + kx,
                                                        k_pmpy + ky)],
        [2 * i * np.conjugate(s) * hopf_x(k_idlx + kx, k_idly + ky) *
         hopf_x(k_sigx + kx, k_sigy + ky),
         2 * (i * np.conjugate(p) + p * np.conjugate(s)) *
         hopf_x(k_idlx + kx, k_idly + ky) * hopf_x(k_pmpx + kx, k_pmpy + ky),
         -omega_i_chosen + enLP(k_idlx + kx, k_idly + ky) - 1j * 1 / 2 * gamma(
             k_idlx + kx, k_idly + ky) + 2 *
         (ni_chosen + np_chosen + ns_chosen) * hopf_x(
             k_idlx + kx, k_idly + ky)**2, (p**2 + 2 * i * s) *
         hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_idlx + kx, k_idly + ky),
         2 * i * p * hopf_x(k_pmpx - kx, k_pmpy - ky) *
         hopf_x(k_idlx + kx, k_idly + ky), i**2 *
         hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_idlx + kx, k_idly + ky)],
        [-(np.conjugate(s)**2 * hopf_x(k_sigx - kx, k_sigy - ky) *
           hopf_x(k_sigx + kx,
                  k_sigy + ky)), -2 * np.conjugate(p) * np.conjugate(s) *
         hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky),
         -((np.conjugate(p)**2 + 2 * np.conjugate(i) * np.conjugate(s)) *
           hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_idlx + kx,
                                                     k_idly + ky)),
         omega_s_chosen - enLP(k_sigx - kx, k_sigy - ky) - 1j * 1 / 2 * gamma(
             k_sigx - kx, k_sigy - ky) - 2 *
         (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_sigx - kx, k_sigy - ky)
         **2, -2 * (i * np.conjugate(p) + p * np.conjugate(s)) * hopf_x(
             k_pmpx - kx, k_pmpy - ky) * hopf_x(
                 k_sigx - kx, k_sigy - ky), -2 * i * np.conjugate(s) * hopf_x(
                     k_idlx - kx, k_idly - ky) * hopf_x(k_sigx - kx,
                                                        k_sigy - ky)],
        [-2 * np.conjugate(p) * np.conjugate(s) *
         hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_sigx + kx, k_sigy + ky),
         -((np.conjugate(p)**2 + 2 * np.conjugate(i) * np.conjugate(s)) *
           hopf_x(k_pmpx - kx, k_pmpy - ky) *
           hopf_x(k_pmpx + kx,
                  k_pmpy + ky)), -2 * np.conjugate(i) * np.conjugate(p) *
         hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_idlx + kx, k_idly + ky),
         -2 * (p * np.conjugate(i) + s * np.conjugate(p)) * hopf_x(
             k_pmpx - kx, k_pmpy - ky) * hopf_x(k_sigx - kx, k_sigy - ky),
         omega_p_chosen - enLP(k_pmpx - kx, k_pmpy - ky) - 1j * 1 / 2 * gamma(
             k_pmpx - kx, k_pmpy - ky) - 2 *
         (ni_chosen + np_chosen + ns_chosen) * hopf_x(
             k_pmpx - kx, k_pmpy - ky)**2, -2 * (
                 i * np.conjugate(p) + p * np.conjugate(s)) * hopf_x(
                     k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx - kx,
                                                        k_pmpy - ky)],
        [-((np.conjugate(p)**2 + 2 * np.conjugate(i) * np.conjugate(s)) *
           hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_sigx + kx,
                                                     k_sigy + ky)), -2 *
         np.conjugate(i) * np.conjugate(p) * hopf_x(k_idlx - kx, k_idly - ky) *
         hopf_x(k_pmpx + kx, k_pmpy + ky), -(
             np.conjugate(i)**2 * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(
                 k_idlx + kx, k_idly + ky)), -2 * s * np.conjugate(i) *
         hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_sigx - kx, k_sigy - ky),
         -2 * (p * np.conjugate(i) + s * np.conjugate(p)) *
         hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx - kx, k_pmpy - ky),
         omega_i_chosen - enLP(k_idlx - kx, k_idly - ky) - 1j * 1 / 2 * gamma(
             k_idlx - kx, k_idly - ky) - 2 *
         (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_idlx - kx, k_idly -
                                                      ky)**2]
    ])


def L_mats(K_X, K_Y, n_kx, n_ky):
    mats = L(K_X, K_Y)
    new_mats = np.transpose(mats, (2, 3, 0, 1))
    new_mats.shape = (n_kx * n_ky, 6, 6)
    return new_mats


def eigL_mats(mats, n_kx, n_ky):
    res = np.linalg.eigvals(mats)
    res.shape = (n_ky, n_kx, 6)
    return res




def eqs(ip):
    #x1 -> omega_s
    #x2 -> ns
    #x3 -> np

    eq1 = "{0:+.16f}*x1{1:+.16f}*x2{2:+.16f}*x3{3:+.16f};".format(
        (-gi - gs) / (gi * xs**2), -alpha**2 + 1, -2 * alpha + 2,
        (-ei * gs + es * gi + 2 * gs * omega_p_chosen) / (gi * xs**2))
    eq2 = ("{0:.16f}*x1^2{1:+.16f}*x1*x2{2:+.16f}*x1*x3{3:+.16f}*x1{4:+.16f}"
           "*x2^2{5:+.16f}*x2*x3{6:+.16f}*x2{7:+.16f}*x3^2{8:+.16f}*x3{9:+.16f};").format(
        xs**(-4), (-4 * alpha - 2) / xs**2, -4 / xs**2, -2 * es / xs**4,
        4 * alpha**2 + 4 * alpha + 1, 8 * alpha + 4, (4 * alpha * es + 2 * es)
        / xs**2, -alpha + 4, 4 * es / xs**2, (4 * es**2 + gs**2) / (4 * xs**4))
    eq3 = ("{0:.16f}*x1^2*x2^2{1:+.16f}*x1*x2^3{2:+.16f}*x1*x2^2*x3{3:+.16f}*x1*"
           "x2^2{4:+.16f}*x1*x2*x3^2{5:+.16f}*x1*x2*x3{6:+.16f}*x2^4{7:+.16f}*x2^3"
           "*x3{8:+.16f}*x2^3{9:+.16f}*x2^2*x3^2{10:+.16f}*x2^2*x3{11:+.16f}*x2^2{12:+.16f}"
           "*x2*x3^3{13:+.16f}*x2*x3^2{14:+.16f}*x2*x3{15:+.16f}*x3^4{16:+.16f}"
           "*x3^3{17:+.16f}*x3^2{18:+.16f}*x3;").format(
        4.0 / xs**4, 1.0 * (-16.0 * alpha - 8.0) / xs**2,
        1.0 * (8.0 * alpha - 8.0) / xs**2, -8.0 * es / xs**4, 4.0 / xs**2,
        1.0 * (4.0 * ep - 4.0 * omega_p_chosen) / (xp**2 * xs**2),
        16.0 * alpha**2 + 16.0 * alpha + 4.0,
        -16.0 * alpha**2 + 8.0 * alpha + 8.0, 1.0 *
        (16.0 * alpha * es + 8.0 * es) / xs**2, 4.0 * alpha**2 - 16.0 * alpha,
        1.0 * (-8.0 * alpha * ep * xs**2 - 8.0 * alpha * es * xp**2 + 8.0 *
               alpha * omega_p_chosen * xs**2 - 4.0 * ep * xs**2 + 8.0 * es *
               xp**2 + 4.0 * omega_p_chosen * xs**2) / (xp**2 * xs**2),
        1.0 * (4.0 * es**2 + 1.0 * gs**2) / xs**4, 4.0 * alpha - 4.0, 1.0 * (
            4.0 * alpha * ep * xs**2 - 4.0 * alpha * omega_p_chosen * xs**2 -
            4.0 * ep * xs**2 - 4.0 * es * xp**2 + 4.0 * omega_p_chosen * xs**2)
        / (xp**2 * xs**2),
        1.0 * (-4.0 * ep * es + 4.0 * es * omega_p_chosen + 1.0 * gs) /
        (xp**2 * xs**2), 1.00000000000000,
        1.0 * (2.0 * ep - 2.0 * omega_p_chosen) / xp**2, 1.0 * (
            1.0 * ep**2 - 2.0 * ep * omega_p_chosen + 1.0 * omega_p_chosen**2 +
            0.25) / xp**4, -1.0 * ip)
    f = [eq1, eq2, eq3]

    #seeding random number generator for reproducible results
    py2c_set_seed(130683)

    s = solve(f, silent=True)
    ns = len(s)  #number of solutions

    R = []
    for k in range(ns):
        d = strsol2dict(s[k])
        x123 = np.array([d['x1'], d['x2'], d['x3']])
        real = np.real(x123)
        imag = np.fabs(np.imag(x123))
        sol = np.empty((real.size + imag.size, ), dtype=real.dtype)
        sol[0::2] = real
        sol[1::2] = imag
        if np.allclose(imag, np.zeros(3)):
            if real[1] > 1e-7:
                R.append((sol[0], sol[2], sol[4], ip))
    return R


solutions = []
for pmp_int in ipx:
    solutions.append(eqs(pmp_int))

solutions_v2 = [sol for sol in solutions if sol != []]

nsnpip = []
for idx in range(1, 4):
    nsnpip.append(np.array([tple[idx] for lst in solutions_v2
                            for tple in lst]))

ipstart = nsnpip[2][0]

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


fig_mf, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(ipx[ipx < ipstart], n_hom_mf(ipx[ipx < ipstart]), color='black', lw=2)
ax.plot(ipx[ipx >= ipstart],
        n_hom_mf(ipx[ipx >= ipstart]),
        linestyle='dashed',
        color='black', lw=2)
for idx in [0, 1]:
    ax.plot(nsnpip[2],
            nsnpip[idx],
            linestyle='none',
            marker=marker_spi[idx],
            markerfacecolor=color_spi[idx])
ax.plot(nsnpip[2],
        alpha * nsnpip[0],
        linestyle='none',
        marker=marker_spi[2],
        markerfacecolor=color_spi[2])
ax.axvline(x=ip_chosen, color='k', ls='dotted', lw=2)
# ax.fill([ipstabini, ipstabfin, ipstabfin, ipstabini], [0, 0, 1.8, 1.8],
#         'gray',nn
#         alpha=0.5,
#         edgecolor='k')
ax.set_xlim(ipx[0], ipx[-1])
ax.set_xlabel(r'$I_p [\gamma_p^3]$')
ax.set_ylabel(r'$\epsilon_s, \epsilon_p, \epsilon_i [\gamma_p]$')
ax.xaxis.set_ticks([0, 4, 8, 12, 16])
ax.yaxis.set_ticks([0.4, 0.8, 1.2, 1.6])
fig_mf.savefig('fig_response_sp_ks_{0:s}.pdf'.format(ks),
               bbox_inches='tight', pad_inches=0.0, transparent=True)

[(omega_s_chosen, ns_chosen, np_chosen, ip_chosen)] = eqs(ip_chosen)
omega_i_chosen = 2 * omega_p_chosen - omega_s_chosen
ni_chosen = alpha * ns_chosen

energy_spi = [omega_s_chosen, omega_p_chosen, omega_i_chosen]


C_band = enC(KX, KY)
LP = enLP(KX, KY)
UP = enUP(KX, KY)

#kxL, kxR = index_mom(-3.5), index_mom(3.5)
kxL, kxR = index_mom(-0.6), index_mom(0.6)

fig_lp, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(KX[nky / 2, kxL:kxR], C_band[nky / 2, kxL:kxR], color='r', ls = 'dashed', lw = 2)
ax.plot(KX[nky / 2, kxL:kxR], LP[nky / 2, kxL:kxR], 'k', lw=2)
ax.plot(KX[nky / 2, kxL:kxR], UP[nky / 2, kxL:kxR], 'k', lw=2)
ax.axhline(y=0, color='r', ls='dashed', lw = 2) # exciton dispersion

for idx in range(3):
    ax.scatter(momentum_spi[idx, 0],
               energy_spi[idx],
               c=color_spi[idx],
               marker=marker_spi[idx],
               s=100)
#ax.axhline(y=ep + 0.5 * np.sqrt(3), color='r', ls='dashed')

ax.annotate('', xy=(k_idlx, omega_i_chosen), xytext=(k_pmpx, omega_p_chosen), size=20,
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
ax.annotate('', xy=(k_sigx, omega_s_chosen), xytext=(k_pmpx, omega_p_chosen), size=20,
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

ax.text(0.1,0.2,'LP', transform=ax.transAxes, fontsize=18)
ax.text(0.4,0.7,'UP', transform=ax.transAxes, fontsize=18)
ax.text(0.2,0.35,'Exciton', transform=ax.transAxes, fontsize=18)
ax.text(0.15,0.9,'Photon', transform=ax.transAxes, fontsize=18)

ax.set_ylim(-25,50)
ax.set_xlim(kx[kxL], kx[kxR])
ax.set_xlabel(r'$k_x\,[\mu m^{-1}]$')

ax.set_ylabel(r'$\omega-\omega_X(0)\,[\gamma_p]$')
fig_lp.savefig('fig_satellites_ks_{0:s}.pdf'.format(ks),
               bbox_inches='tight', pad_inches=0.0, transparent=True)

p = 1 / np.sqrt(ip_chosen) * (
    2 / xs**2 * (es - 1j * gs / 2 - omega_s_chosen) * ns_chosen + 2 *
    (2 * np_chosen + ns_chosen + 2 * ni_chosen) * ns_chosen -
    (1 / xp**2 * (ep + 1j * 1 / 2 - omega_p_chosen) * np_chosen +
     (np_chosen + 2 * ns_chosen + 2 * ni_chosen) * np_chosen))

pr = p.real
pi = p.imag

matSI = np.array(
    [[2 * ni_chosen + 2 * np_chosen + ns_chosen +
      (es - omega_s_chosen) / xs**2, gs / (2. * xs
                                           **2), -pi**2 + pr**2, 2 * pi * pr],
     [-gs / (2. * xs**2), 2 * ni_chosen + 2 * np_chosen + ns_chosen +
      (es - omega_s_chosen) / xs**2, 2 * pi * pr, pi**2 - pr**2],
     [-pi**2 + pr**2, 2 * pi * pr, ni_chosen + 2 * np_chosen + 2 * ns_chosen +
      (ei - omega_i_chosen) / xi**2,
      gi / (2. * xi**2)], [2 * pi * pr, pi**2 - pr**2, -gi / (2. * xi**2),
                           ni_chosen + 2 * np_chosen + 2 * ns_chosen +
                           (ei - omega_i_chosen) / xi**2]])

norm = ns_chosen + ni_chosen
N = null(matSI) * np.sqrt(norm)

[sr, si, ir, ii] = N[:, 0]
s = sr + 1j * si
i = ir + 1j * ii


matsL = L_mats(KX, KY, nkx, nky)
eigs = eigL_mats(matsL, nkx, nky)

kxL, kxR = index_mom(-0.55), index_mom(0.55)

mpl.rcParams.update({'axes.labelsize': 30,
                     'axes.linewidth': 3,
                     'xtick.labelsize': 25,
                     'ytick.labelsize': 25,
                     'xtick.major.size': 11,
                     'ytick.major.size': 11,
                     'xtick.major.width': 3,
                     'ytick.major.width': 3})

fig_excitation, axes = plt.subplots(1, 2, figsize=(20, 6))
for idx in range(6):
    axes[0].plot(KX[nky / 2, kxL:kxR],
                 eigs[nky / 2, kxL:kxR, idx].real,
                 linestyle='none',
                 marker='o',
                 markerfacecolor='black',
                 markersize=3)
for idx in range(6):
    axes[1].plot(KX[nky / 2, kxL:kxR],
                 eigs[nky / 2, kxL:kxR, idx].imag,
                 linestyle='none',
                 marker='o',
                 markerfacecolor='black',
                 markersize=3)
#ax.axhline(y=0, color='black', ls='dashed', lw=2)
for ax in axes:
    ax.set_xlabel(r'$k_x - k_n\,[\mu m^{-1}]$')
    ax.set_xlim(kx[kxL], kx[kxR])
    ax.xaxis.set_ticks([-0.5, 0, 0.5])

axes[0].text(0.5,0.52,'G', transform=axes[0].transAxes, fontsize=30)
axes[1].text(0.5,0.9,'G', transform=axes[1].transAxes, fontsize=30)

    
axes[0].set_ylim(-5, 5)
axes[0].set_ylabel(r'$\Re{(\omega)}\,[\gamma_p]$')

axes[1].set_ylim(-1, 0)
axes[1].set_ylabel(r'$\Im{(\omega)}\,[\gamma_p]$')

fig_excitation.savefig('fig_response_ev_ks_{0:s}.pdf'.format(ks),
               bbox_inches='tight', pad_inches=0.0, transparent=True)
