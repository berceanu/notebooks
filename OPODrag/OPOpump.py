# Mean-field: pump only

import numpy as np
from scipy import optimize
from phcpy.phcpy2c import py2c_set_seed
from phcpy.solver import solve
from phcpy.solutions import strsol2dict
import ConfigParser
import sys

def read_parameters(filename):
    config = ConfigParser.RawConfigParser()
    try:
        config.readfp(open(filename))
    except IOError:
        raise IOError('cannot find parameters.ini, exiting')
    return config

param = read_parameters("/home/berceanu/notebooks/OPODrag/parameters.ini")
param_set = sys.argv[1]

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
delta_k = param.getfloat(param_set, "delta_k")

k_idlx = 2 * k_pmpx - k_sigx
k_idly = 2 * k_pmpy - k_sigy

gp = gamma_C + (1 / np.sqrt(1 + (Omega_R / ((0.5 * ((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)) + omega_X) - 0.5 * np.sqrt(((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)) - omega_X) ** 2 + 4 * Omega_R ** 2)) - (omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)))) ** 2)) ** 2 * (gamma_X - gamma_C)
omega_p_chosen = (omega_pmp - omega_X) / gp

side_k = nkx * delta_k / 2
side_r = np.pi / delta_k
delta_r = np.pi / side_k

x = y = np.arange(-side_r, side_r, delta_r)
kx = ky = np.arange(-side_k, side_k, delta_k)

KX, KY = np.meshgrid(kx, ky)
X, Y = np.meshgrid(x, y)

ipx = np.linspace(ipx_start, ipx_end, 30)

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
                                                 + xp ** 2 * n) ** 2 + 1./4.) * n - xp ** 4 * ip, 0, 3)
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

    #x1 -> omega_s
    #x2 -> ns
    #x3 -> np
       
    eq1 = "{0:+.16f}*x1{1:+.16f}*x2{2:+.16f}*x3{3:+.16f};"          .format((-gi - gs)/(gi*xs**2), -alpha**2 + 1, -2*alpha + 2, (-ei*gs + es*gi + 2*gs*omega_p_chosen)/(gi*xs**2))
    eq2 = "{0:.16f}*x1^2{1:+.16f}*x1*x2{2:+.16f}*x1*x3{3:+.16f}*x1{4:+.16f}*x2^2{5:+.16f}*x2*x3{6:+.16f}*x2{7:+.16f}*x3^2{8:+.16f}*x3{9:+.16f};"          .format(xs**(-4), (-4*alpha - 2)/xs**2, -4/xs**2, -2*es/xs**4, 4*alpha**2 + 4*alpha + 1, 8*alpha + 4, (4*alpha*es + 2*es)/xs**2, -alpha + 4, 4*es/xs**2, (4*es**2 + gs**2)/(4*xs**4))
    eq3 = "{0:.16f}*x1^2*x2^2{1:+.16f}*x1*x2^3{2:+.16f}*x1*x2^2*x3{3:+.16f}*x1*x2^2{4:+.16f}*x1*x2*x3^2{5:+.16f}*x1*x2*x3{6:+.16f}*x2^4{7:+.16f}*x2^3*x3{8:+.16f}*x2^3{9:+.16f}*x2^2*x3^2{10:+.16f}*x2^2*x3{11:+.16f}*x2^2{12:+.16f}*x2*x3^3{13:+.16f}*x2*x3^2{14:+.16f}*x2*x3{15:+.16f}*x3^4{16:+.16f}*x3^3{17:+.16f}*x3^2{18:+.16f}*x3;"          .format(4.0/xs**4, 1.0*(-16.0*alpha - 8.0)/xs**2, 1.0*(8.0*alpha - 8.0)/xs**2, -8.0*es/xs**4, 4.0/xs**2, 1.0*(4.0*ep - 4.0*omega_p_chosen)/(xp**2*xs**2), 16.0*alpha**2 + 16.0*alpha + 4.0, -16.0*alpha**2 + 8.0*alpha + 8.0, 1.0*(16.0*alpha*es + 8.0*es)/xs**2, 4.0*alpha**2 - 16.0*alpha, 1.0*(-8.0*alpha*ep*xs**2 - 8.0*alpha*es*xp**2 + 8.0*alpha*omega_p_chosen*xs**2 - 4.0*ep*xs**2 + 8.0*es*xp**2 + 4.0*omega_p_chosen*xs**2)/(xp**2*xs**2), 1.0*(4.0*es**2 + 1.0*gs**2)/xs**4, 4.0*alpha - 4.0, 1.0*(4.0*alpha*ep*xs**2 - 4.0*alpha*omega_p_chosen*xs**2 - 4.0*ep*xs**2 - 4.0*es*xp**2 + 4.0*omega_p_chosen*xs**2)/(xp**2*xs**2), 1.0*(-4.0*ep*es + 4.0*es*omega_p_chosen + 1.0*gs)/(xp**2*xs**2), 1.00000000000000, 1.0*(2.0*ep - 2.0*omega_p_chosen)/xp**2, 1.0*(1.0*ep**2 - 2.0*ep*omega_p_chosen + 1.0*omega_p_chosen**2 + 0.25)/xp**4, -1.0*ip)
    f = [eq1, eq2, eq3]
    
    #seeding random number generator for reproducible results
    py2c_set_seed(130683)
    
    s = solve(f,silent=True)
    ns = len(s) #number of solutions

    R = []
    for k in range(ns):
        d = strsol2dict(s[k])
        x123 = np.array([d['x1'], d['x2'], d['x3']])
        real = np.real(x123)
        imag = np.fabs(np.imag(x123))
        sol = np.empty((real.size + imag.size,), dtype=real.dtype)
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
    nsnpip.append(np.array([tple[idx]
                  for lst in solutions_v2 for tple in lst]))


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
s = 0.
i = 0.

matsL = L_mats(KX, KY, nkx, nky)
vectfd = fd_mats(KX, KY, nkx, nky)
bcoef = bog_coef_mats(matsL, vectfd, nkx, nky)

matsL_min = -np.fft.fftshift(matsL, axes=(1, 2))
vectfd_min = -np.fft.fftshift(vectfd, axes=(1,))
bcoef_conj_mink = bog_coef_mats(matsL_min, vectfd_min, nkx, nky)

psi_k = gv / 2 * (bcoef[:, :, 0:3] + bcoef_conj_mink[:, :, 3:6])

psi_k[nky / 2, nkx / 2, :] += np.sqrt(nkx * nky) * \
    np.array([s / xs, p / xp, i / xi])

psi_r = np.fft.fftshift(
    np.fft.ifft2(np.sqrt(nkx * nky) * psi_k, axes=(0, 1)), axes=(0, 1))
res_r = np.abs(psi_r) ** 2 / \
    np.array([ns_chosen / xs ** 2, np_chosen / xp ** 2, ni_chosen / xi ** 2])


np.save("/home/berceanu/notebooks/OPODrag/res_r", res_r)
print "done!"
