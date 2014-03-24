import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
import phcpy2c as phc
import ConfigParser

def read_parameters(filename):
    config = ConfigParser.RawConfigParser()
    try:
        config.readfp(open(filename))
    except IOError:
        raise IOError('cannot find parameters.ini, exiting')
    return config

param = read_parameters("/home/berceanu/Envs/env-ipython/notebooks/parameters.ini")
#param_set = "ks-0.4"
param_set = "ks0.0"
#param_set = "ks0.7"

omega_X = param.getfloat(param_set, "omega_X")
omega_C0 = param.getfloat(param_set, "omega_C0")
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

nkx = param.getfloat(param_set, "nkx")
nky = param.getfloat(param_set, "nky")
ipx_start = param.getfloat(param_set, "ipx_start")
ipx_end = param.getfloat(param_set, "ipx_end")
kxl = param.getfloat(param_set, "kxl")
kxr = param.getfloat(param_set, "kxr")
kyl = param.getfloat(param_set, "kyl")
kyr = param.getfloat(param_set, "kyr")
eigs_threshold = param.getfloat(param_set, "eigs_threshold")
stride_r = param.getint(param_set, "stride_r")
stride_c = param.getint(param_set, "stride_c")


k_idlx = 2 * k_pmpx - k_sigx
k_idly = 2 * k_pmpy - k_sigy
ks = ("{0:.3f}".format(k_sigx)).replace(".", "_")

gp = gamma_C + (1 / np.sqrt(1 + (Omega_R / ((0.5 * ((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)) + omega_X) - 0.5 * np.sqrt(((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)) - omega_X) ** 2 + 4 * Omega_R ** 2)) - (omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)))) ** 2)) ** 2 * (gamma_X - gamma_C)
omega_p_chosen = (omega_pmp - omega_X) / gp

delta_k = 0.05

side_k = nkx * delta_k / 2
side_r = np.pi / delta_k
delta_r = np.pi / side_k

x = y = np.arange(-side_r, side_r, delta_r)
kx = ky = np.arange(-side_k, side_k, delta_k)

KX, KY = np.meshgrid(kx, ky)
X, Y = np.meshgrid(x, y)

ipx = np.linspace(ipx_start, ipx_end, 30)

x_label_k = r'$k_x[\mu m^{-1}]$'
y_label_k = r'$k_y[\mu m^{-1}]$'
x_label_i = r'$x[\mu m]$'
y_label_i = r'$y[\mu m]$'

letter_spi = ['s', 'p', 'i']
title_k = []
for idx in range(3):
    title_k.append(r'$g \left|\tilde{\Psi}_{LP}^{' + letter_spi[idx] +
                   r'}\left(k+k_{' + letter_spi[idx] +
                   r'}\right)\right|^{2} [\gamma_p \mu m^4]$')
title_i = []
for idx in range(3):
    title_i.append(r'$I^' + letter_spi[idx] + r'$')

color_spi = ['blue', 'red', 'green']

momentum_spi = np.array([[k_sigx, k_sigy],
                         [k_pmpx, k_pmpy],
                         [k_idlx, k_idly]])


def null(A, eps=1e-10):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def index_mom(mom):
    return int(np.floor((mom + side_k) / delta_k))


def enC(kx, ky):
    return (omega_C0 * np.sqrt(1 + (np.sqrt(kx ** 2 + ky ** 2) / kz) ** 2)
            - omega_X) / gp

def enLP(kx, ky):
    return 0.5 * enC(kx, ky) - 0.5 * np.sqrt(enC(kx, ky) ** 2
                                             + 4 * Omega_R ** 2 / gp ** 2)

def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + ((Omega_R / gp)
                            / (enLP(kx, ky) - enC(kx, ky))) ** 2)

def blue_enLP(kx, ky):
    return enLP(kx, ky) + 2 * hopf_x(kx, ky) ** 2 *\
        (ni_chosen + np_chosen + ns_chosen)

def hopf_c(kx, ky):
    return -1 / np.sqrt(1 + ((enLP(kx, ky) - enC(kx, ky))
                             / (Omega_R / gp)) ** 2)


def gamma(kx, ky):
    return (gamma_C + hopf_x(kx, ky) ** 2 * (gamma_X - gamma_C)) / gp


kxl, kxr = index_mom(kxl), index_mom(kxr)
kyl, kyr = index_mom(kyl), index_mom(kyr)

es = enLP(k_sigx, k_sigy)
ep = enLP(k_pmpx, k_pmpy)
ei = enLP(k_idlx, k_idly)

gs = gamma(k_sigx, k_sigy)
gi = gamma(k_idlx, k_idly)

xs = hopf_x(k_sigx, k_sigy)
xp = hopf_x(k_pmpx, k_pmpy)
xi = hopf_x(k_idlx, k_idly)

cs = hopf_c(k_sigx, k_sigy)
cp = hopf_c(k_pmpx, k_pmpy)
ci = hopf_c(k_idlx, k_idly)

alpha = (xi ** 2 / xs ** 2) * (gs / gi)

def n_hom_mf(ips):
    return np.array([optimize.brentq(lambda n: ((ep - omega_p_chosen
                                                 + xp ** 2 * n) ** 2 + 1 / 4) * n - xp ** 4 * ip, 0, 3)
                     for ip in ips])


def L(kx, ky):
    return np.array(
        [[-omega_s_chosen + enLP(
            k_sigx + kx, k_sigy + ky) - 1j * 1 / 2 * gamma(
            k_sigx + kx, k_sigy + ky) + 2 * (
            ni_chosen + np_chosen + ns_chosen) * hopf_x(
            k_sigx + kx, k_sigy + ky) ** 2, 2 * (
            p * np.conjugate(
                i) + s * np.conjugate(
                p)) * hopf_x(
            k_pmpx + kx, k_pmpy + ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), 2 * s * np.conjugate(
            i) * hopf_x(
            k_idlx + kx, k_idly + ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), s ** 2 * hopf_x(
            k_sigx - kx, k_sigy - ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), 2 * p * s * hopf_x(
            k_pmpx - kx, k_pmpy - ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), (
            p ** 2 + 2 * i * s) * hopf_x(
            k_idlx - kx, k_idly - ky) * hopf_x(
            k_sigx + kx, k_sigy + ky)],
         [2 * (i * np.conjugate(p) + p * np.conjugate(s)) * hopf_x(k_pmpx + kx, k_pmpy + ky) * hopf_x(k_sigx + kx, k_sigy + ky), -omega_p_chosen + enLP(k_pmpx + kx, k_pmpy + ky) - 1j * 1 / 2 * gamma(k_pmpx + kx, k_pmpy + ky) + 2 * (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_pmpx + kx, k_pmpy + ky) ** 2, 2 * (p * np.conjugate(i) + s * np.conjugate(p))
          * hopf_x(
              k_idlx + kx, k_idly + ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky), 2 * p * s * hopf_x(
              k_sigx - kx, k_sigy - ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky), (
              p ** 2 + 2 * i * s) * hopf_x(
              k_pmpx - kx, k_pmpy - ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky), 2 * i * p * hopf_x(
              k_idlx - kx, k_idly - ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky)],
         [2 * i * np.conjugate(s) * hopf_x(k_idlx + kx, k_idly + ky) * hopf_x(k_sigx + kx, k_sigy + ky), 2 * (i * np.conjugate(p) + p * np.conjugate(s)) * hopf_x(k_idlx + kx, k_idly + ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), -omega_i_chosen + enLP(k_idlx + kx, k_idly + ky) - 1j * 1 / 2 * gamma(k_idlx + kx, k_idly + ky) + 2 *
          (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_idlx + kx, k_idly + ky) ** 2, (p ** 2 + 2 * i * s) * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_idlx + kx, k_idly + ky), 2 * i * p * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_idlx + kx, k_idly + ky), i ** 2 * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_idlx + kx, k_idly + ky)],
         [-(np.conjugate(s) ** 2 * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_sigx + kx, k_sigy + ky)), -2 * np.conjugate(p) * np.conjugate(s) * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), -((np.conjugate(p) ** 2 + 2 * np.conjugate(i) * np.conjugate(s)) * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_idlx + kx, k_idly + ky)), omega_s_chosen -
          enLP(
              k_sigx - kx, k_sigy - ky) - 1j * 1 / 2 * gamma(
              k_sigx - kx, k_sigy - ky) - 2 * (
              ni_chosen + np_chosen + ns_chosen) * hopf_x(
              k_sigx - kx, k_sigy - ky) ** 2, -2 * (
              i * np.conjugate(
                  p) + p * np.conjugate(
                  s)) * hopf_x(
              k_pmpx - kx, k_pmpy - ky) * hopf_x(
              k_sigx - kx, k_sigy - ky), -2 * i * np.conjugate(
              s) * hopf_x(
              k_idlx - kx, k_idly - ky) * hopf_x(
              k_sigx - kx, k_sigy - ky)],
         [-2 * np.conjugate(p) * np.conjugate(s) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_sigx + kx, k_sigy + ky), -((np.conjugate(p) ** 2 + 2 * np.conjugate(i) * np.conjugate(s)) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky)), -2 * np.conjugate(i) * np.conjugate(p) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_idlx + kx, k_idly + ky), -2 * (p * np.conjugate(i) + s * np.conjugate(p))
          * hopf_x(
              k_pmpx - kx, k_pmpy - ky) * hopf_x(
              k_sigx - kx, k_sigy - ky), omega_p_chosen - enLP(
              k_pmpx - kx, k_pmpy - ky) - 1j * 1 / 2 * gamma(
              k_pmpx - kx, k_pmpy - ky) - 2 * (
              ni_chosen + np_chosen + ns_chosen) * hopf_x(
              k_pmpx - kx, k_pmpy - ky) ** 2, -2 * (
              i * np.conjugate(
                  p) + p * np.conjugate(
                  s)) * hopf_x(
              k_idlx - kx, k_idly - ky) * hopf_x(
              k_pmpx - kx, k_pmpy - ky)],
         [-((np.conjugate(p) ** 2 + 2 * np.conjugate(i) * np.conjugate(s)) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_sigx + kx, k_sigy + ky)), -2 * np.conjugate(i) * np.conjugate(p) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), -(np.conjugate(i) ** 2 * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_idlx + kx, k_idly + ky)), -2 * s * np.conjugate(i) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_sigx - kx, k_sigy - ky), -2 * (p * np.conjugate(i) + s * np.conjugate(p)) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx - kx, k_pmpy - ky), omega_i_chosen - enLP(k_idlx - kx, k_idly - ky) - 1j * 1 / 2 * gamma(k_idlx - kx, k_idly - ky) - 2 * (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_idlx - kx, k_idly - ky) ** 2]])


def L_mats(K_X, K_Y, n_kx, n_ky):
    mats = L(K_X, K_Y)
    new_mats = np.transpose(mats, (2, 3, 0, 1))
    new_mats.shape = (n_kx * n_ky, 6, 6)
    return new_mats


def eigL_mats(mats, n_kx, n_ky):
    res = np.linalg.eigvals(mats)
    res.shape = (n_ky, n_kx, 6)
    return res


def fd(kx, ky):
    return np.array([cs / xs * hopf_c(kx + k_sigx, ky + k_sigy) * s,
                     cp / xp * hopf_c(kx + k_pmpx, ky + k_pmpy) * p,
                     ci / xi * hopf_c(kx + k_idlx, ky + k_idly) * i,
                    -cs / xs * hopf_c(k_sigx - kx, k_sigy - ky) *
                     np.conjugate(s),
                    -cp / xp * hopf_c(k_pmpx - kx, k_pmpy - ky) *
                     np.conjugate(p),
                    -ci / xi * hopf_c(k_idlx - kx, k_idly - ky)
                    * np.conjugate(i)])


def fd_mats(K_X, K_Y, n_kx, n_ky):
    res = fd(K_X, K_Y)
    new_res = np.transpose(res, (1, 2, 0))
    new_res.shape = (n_kx * n_ky, 6)
    return new_res


def bog_coef_mats(mats, fds, n_kx, n_ky):
    res = np.linalg.solve(mats, -fds)
    res.shape = (n_ky, n_kx, 6)
    return res


def eqs(ip):

    # x1 -> omega_s
    # x2 -> ns
    # x3 -> np

    eq1 = "{0:+.16f}*x1{1:+.16f}*x2{2:+.16f}*x3{3:+.16f};"\
          .format((-gi - gs) / (gi * xs ** 2), -alpha ** 2 + 1, -2 * alpha + 2, (-ei * gs + es * gi + 2 * gs * omega_p_chosen) / (gi * xs ** 2))
    eq2 = "{0:.16f}*x1^2{1:+.16f}*x1*x2{2:+.16f}*x1*x3{3:+.16f}*x1{4:+.16f}*x2^2{5:+.16f}*x2*x3{6:+.16f}*x2{7:+.16f}*x3^2{8:+.16f}*x3{9:+.16f};"\
          .format(xs ** (-4), (-4 * alpha - 2) / xs ** 2, -4 / xs ** 2, -2 * es / xs ** 4, 4 * alpha ** 2 + 4 * alpha + 1, 8 * alpha + 4, (4 * alpha * es + 2 * es) / xs ** 2, -alpha + 4, 4 * es / xs ** 2, (4 * es ** 2 + gs ** 2) / (4 * xs ** 4))
    eq3 = "{0:.16f}*x1^2*x2^2{1:+.16f}*x1*x2^3{2:+.16f}*x1*x2^2*x3{3:+.16f}*x1*x2^2{4:+.16f}*x1*x2*x3^2{5:+.16f}*x1*x2*x3{6:+.16f}*x2^4{7:+.16f}*x2^3*x3{8:+.16f}*x2^3{9:+.16f}*x2^2*x3^2{10:+.16f}*x2^2*x3{11:+.16f}*x2^2{12:+.16f}*x2*x3^3{13:+.16f}*x2*x3^2{14:+.16f}*x2*x3{15:+.16f}*x3^4{16:+.16f}*x3^3{17:+.16f}*x3^2{18:+.16f}*x3;"\
          .format(4.0 / xs ** 4, 1.0 * (-16.0 * alpha - 8.0) / xs ** 2, 1.0 * (8.0 * alpha - 8.0) / xs ** 2, -8.0 * es / xs ** 4, 4.0 / xs ** 2, 1.0 * (4.0 * ep - 4.0 * omega_p_chosen) / (xp ** 2 * xs ** 2), 16.0 * alpha ** 2 + 16.0 * alpha + 4.0, -16.0 * alpha ** 2 + 8.0 * alpha + 8.0, 1.0 * (16.0 * alpha * es + 8.0 * es) / xs ** 2, 4.0 * alpha ** 2 - 16.0 * alpha, 1.0 * (-8.0 * alpha * ep * xs ** 2 - 8.0 * alpha * es * xp ** 2 + 8.0 * alpha * omega_p_chosen * xs ** 2 - 4.0 * ep * xs ** 2 + 8.0 * es * xp ** 2 + 4.0 * omega_p_chosen * xs ** 2) / (xp ** 2 * xs ** 2), 1.0 * (4.0 * es ** 2 + 1.0 * gs ** 2) / xs ** 4, 4.0 * alpha - 4.0, 1.0 * (4.0 * alpha * ep * xs ** 2 - 4.0 * alpha * omega_p_chosen * xs ** 2 - 4.0 * ep * xs ** 2 - 4.0 * es * xp ** 2 + 4.0 * omega_p_chosen * xs ** 2) / (xp ** 2 * xs ** 2), 1.0 * (-4.0 * ep * es + 4.0 * es * omega_p_chosen + 1.0 * gs) / (xp ** 2 * xs ** 2), 1.00000000000000, 1.0 * (2.0 * ep - 2.0 * omega_p_chosen) / xp ** 2, 1.0 * (1.0 * ep ** 2 - 2.0 * ep * omega_p_chosen + 1.0 * omega_p_chosen ** 2 + 0.25) / xp ** 4, -1.0 * ip)

    phc.py2c_syscon_clear_system()
    phc.py2c_solcon_clear_solutions()
    phc.py2c_syscon_initialize_number(3)

    phc.py2c_syscon_store_polynomial(len(eq1), 3, 1, eq1)
    phc.py2c_syscon_store_polynomial(len(eq2), 3, 2, eq2)
    phc.py2c_syscon_store_polynomial(len(eq3), 3, 3, eq3)

    phc.py2c_solve_system()
    ns = phc.py2c_solcon_number_of_solutions()

    R = []
    for k in range(1, ns + 1):
        out = []
        out = phc.py2c_solcon_retrieve_solution(4, k)
        sol = [elem for tple in out[:-1] for elem in tple]
        imag = [np.fabs(elem) for elem in sol[1::2]]
        real = np.array([elem for elem in sol[::2]])
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
    nsnpip.append(np.array([tple[idx]
                  for lst in solutions_v2 for tple in lst]))


fig_mf, ax = plt.subplots()
ax.plot(ipx, n_hom_mf(ipx), linestyle='none',
        marker='o', markerfacecolor='black')
for idx in [0, 1]:
    ax.plot(nsnpip[2], nsnpip[idx], linestyle='none',
            marker='o', markerfacecolor=color_spi[idx])
ax.plot(nsnpip[2], alpha * nsnpip[0], linestyle='none',
        marker='o', markerfacecolor=color_spi[2])
ax.axvline(x=ip_chosen, color='k', ls='dashed')
ax.set_xlim(ipx[0], ipx[-1])
ax.set_xlabel(r'$I_p [\gamma_p^3]$')
ax.set_ylabel(r'$n_s, n_p [\gamma_p]$')
plt.close(fig_mf)

[(omega_s_chosen, ns_chosen, np_chosen, ip_chosen)] = eqs(ip_chosen)
omega_i_chosen = 2 * omega_p_chosen - omega_s_chosen
ni_chosen = alpha * ns_chosen


energy_spi = [omega_s_chosen, omega_p_chosen, omega_i_chosen]


p = 1 / np.sqrt(ip_chosen) * (2 / xs ** 2 * (es - 1j * gs / 2 - omega_s_chosen) * ns_chosen + 2 * (2 * np_chosen + ns_chosen + 2 * ni_chosen)
                              * ns_chosen - (1 / xp ** 2 * (ep + 1j * 1 / 2 - omega_p_chosen) * np_chosen + (np_chosen + 2 * ns_chosen + 2 * ni_chosen) * np_chosen))

pr = p.real
pi = p.imag


matSI = np.array(
    [[2 * ni_chosen + 2 * np_chosen + ns_chosen + (
        es - omega_s_chosen) / xs ** 2, gs / (
        2. * xs ** 2), -pi ** 2 + pr ** 2, 2 * pi * pr],
     [-gs / (2. * xs ** 2), 2 * ni_chosen + 2 * np_chosen + ns_chosen +
      (es - omega_s_chosen) / xs ** 2, 2 * pi * pr, pi ** 2 - pr ** 2],
     [-pi ** 2 + pr ** 2, 2 * pi * pr, ni_chosen + 2 * np_chosen +
      2 * ns_chosen + (ei - omega_i_chosen) / xi ** 2, gi / (2. * xi ** 2)],
     [2 * pi * pr, pi ** 2 - pr ** 2, -gi / (2. * xi ** 2), ni_chosen + 2 * np_chosen + 2 * ns_chosen + (ei - omega_i_chosen) / xi ** 2]])


norm = ns_chosen + ni_chosen
N = null(matSI) * np.sqrt(norm)

[sr, si, ir, ii] = N[:, 0]
s = i = 0

matsL = L_mats(KX, KY, nkx, nky)
vectfd = fd_mats(KX, KY, nkx, nky)
bcoef = bog_coef_mats(matsL, vectfd, nkx, nky)

matsL_min = -np.fft.fftshift(matsL, axes=(1, 2))
vectfd_min = -np.fft.fftshift(vectfd, axes=(1,))
bcoef_conj_mink = bog_coef_mats(matsL_min, vectfd_min, nkx, nky)

psi_k = gv / 2 * (bcoef[:, :, 0:3] + bcoef_conj_mink[:, :, 3:6])


[imax, jmax] = np.unravel_index(
    np.argmax(np.abs(psi_k[:, :, 0])), psi_k[:, :, 0].shape)


l1 = psi_k[imax - 2:imax + 3, jmax - 2, :]
l2 = psi_k[imax - 2:imax + 3, jmax + 2, :]
l3 = psi_k[imax - 2, jmax - 1:jmax + 2, :]
l4 = psi_k[imax + 2, jmax - 1:jmax + 2, :]

l = np.concatenate((l1, l2, l3, l4))
averages = np.mean(l, axis=0)
psi_k[imax - 1:imax + 2, jmax - 1:jmax + 2, :] = averages


psi_k[nky / 2, nkx / 2, :] += np.sqrt(nkx * nky) * \
    np.array([s / xs, p / xp, i / xi])


res_k = np.log10(np.abs(psi_k) ** 2)  # logscale


fig_mom_S, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(res_k[kyl:kyr, kxl:kxr, 0],
                 cmap=cm.gray, origin=None, extent=[kx[kxl], kx[kxr], ky[kyl], ky[kyr]])
axes[1].plot(kx[kxl:kxr], res_k[nky / 2, kxl:kxr, 0]) 
axes[1].grid()
axes[0].set_ylabel(y_label_k)
axes[0].set_xlabel(x_label_k)
axes[1].set_xlabel(x_label_k)
plt.close(fig_mom_S)

fig_mom_P, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(res_k[kyl:kyr, kxl:kxr, 1],
                 cmap=cm.gray, origin=None, extent=[kx[kxl], kx[kxr], ky[kyl], ky[kyr]])
axes[1].plot(kx[kxl:kxr], res_k[nky / 2, kxl:kxr, 1]) 
axes[1].grid()
axes[0].set_ylabel(y_label_k)
axes[0].set_xlabel(x_label_k)
axes[1].set_xlabel(x_label_k)
plt.close(fig_mom_P)

fig_mom_I, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(res_k[kyl:kyr, kxl:kxr, 2],
                 cmap=cm.gray, origin=None, extent=[kx[kxl], kx[kxr], ky[kyl], ky[kyr]])
axes[1].plot(kx[kxl:kxr], res_k[nky / 2, kxl:kxr, 2]) 
axes[1].grid()
axes[0].set_ylabel(y_label_k)
axes[0].set_xlabel(x_label_k)
axes[1].set_xlabel(x_label_k)
plt.close(fig_mom_I)


psi_r = np.fft.fftshift(
    np.fft.ifft2(np.sqrt(nkx * nky) * psi_k, axes=(0, 1)), axes=(0, 1))
res_r = np.abs(psi_r) ** 2 / \
    np.array([ns_chosen / xs ** 2, np_chosen / xp ** 2, ni_chosen / xi ** 2])

rango = 200 / (1024/nkx)

fig_real_S, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(res_r[rango:-rango, rango:-rango, 0], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])
axes[1].plot(x[rango:-rango], res_r[nky / 2, rango:-rango, 0])
axes[1].grid()
axes[0].set_ylabel(y_label_i)
axes[0].set_xlabel(x_label_i)
axes[1].set_xlabel(x_label_i)
plt.close(fig_real_S)

fig_real_P, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(res_r[rango:-rango, rango:-rango, 1], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])
axes[1].plot(x[rango:-rango], res_r[nky / 2, rango:-rango, 1])
axes[1].grid()
axes[0].set_ylabel(y_label_i)
axes[0].set_xlabel(x_label_i)
axes[1].set_xlabel(x_label_i)
plt.close(fig_real_P)

fig_real_I, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(res_r[rango:-rango, rango:-rango, 2], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])
axes[1].plot(x[rango:-rango], res_r[nky / 2, rango:-rango, 2])
axes[1].grid()
axes[0].set_ylabel(y_label_i)
axes[0].set_xlabel(x_label_i)
axes[1].set_xlabel(x_label_i)
plt.close(fig_real_I)

print 'done!'
