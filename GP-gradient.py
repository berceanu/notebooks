import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

lC = 0.868

Omega_R = 4.4 / 2 * 10 ** (-3)
omega_C0 = omega_X = 1.53

energy_spi = np.array([-0.748, -0.2, 0.349])

k_sigx = - 0.179
k_pmpx = 1.391
k_idlx = 2.962
k_sigy = 0.
k_pmpy = 0.
k_idly = 0.
momentum_spi = np.array([[k_sigx, k_sigy],
                         [k_pmpx, k_pmpy],
                         [k_idlx, k_idly]])

kz = 1 / lC * np.sqrt(omega_C0 / (2 * Omega_R))

def enC(kx, ky):
    return omega_C0 * np.sqrt(1 + (np.sqrt(kx ** 2 + ky ** 2) / kz) ** 2)

def enLP(kx, ky):
    return (enC(kx, ky) + omega_X) / 2 - np.sqrt(Omega_R ** 2 + ((enC(kx, ky) - omega_X) / 2) ** 2)

def enUP(kx, ky):
    return (enC(kx, ky) + omega_X) / 2 + np.sqrt(Omega_R ** 2 + ((enC(kx, ky) - omega_X) / 2) ** 2)

def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + (Omega_R / (enLP(kx, ky) - enC(kx, ky))) ** 2)

gamma_X = 2 * 0.12 * Omega_R
gamma_C = 2 * 0.12 * Omega_R
def gamma(kx, ky):
    return gamma_C + hopf_x(kx, ky) ** 2 * (gamma_X - gamma_C)
gp = gamma(k_pmpx, k_pmpy)

def find_index(dist_or_mom, side, delta):
    return int(np.around((side + dist_or_mom) / delta))

x_label = r'$x[\mu m]$'
y_label = r'$y[\mu m]$'
titles = ['signal', 'pump', 'idler']
color_spi = ['blue', 'red', 'green']

Ny = 256
Ly = 70.
y, a_y = np.linspace(-Ly, Ly, num=Ny, endpoint=False, retstep=True)
Nx = 256
Lx = 70.
x, a_x = np.linspace(-Lx, Lx, num=Nx, endpoint=False, retstep=True)
Nt = 2048
dxsav_sp = 1.
t, delta_t = np.linspace(0, Nt - 1, num=Nt, endpoint=False, retstep=True)

side_omega = dxsav_sp * np.pi / delta_t
omega, delta_omega = np.linspace(-side_omega, side_omega, num=Nt, endpoint=False, retstep=True)

idx_spi = np.zeros((3,), dtype=np.int)
for idx in range(3):
    idx_spi[idx] = find_index(energy_spi[idx], side_omega, delta_omega)

base = '/home/berceanu/data/'
folder = r'OPO/tophat/defect/kp/1.4/fp/0.11/N/256/move_defect/spectrum_2048/'
path_s = base + folder + r'opo_ph-spc_enfilt_signal.dat'
path_p = base + folder + r'opo_ph-spc_enfilt_pump.dat'
path_i = base + folder + r'opo_ph-spc_enfilt_idler.dat'
path_spect = base + folder + r'spectr_om-vs-kx_no-trigg.dat'
path_spect_int = base + folder + r'int-spectr_no-trigg.dat'

data_s = np.loadtxt(path_s, usecols=(2,))
data_p = np.loadtxt(path_p, usecols=(2,))
data_i = np.loadtxt(path_i, usecols=(2,))

data_spect = np.loadtxt(path_spect, usecols=(2,))
data_spect_int = np.loadtxt(path_spect_int, usecols=(1,))

data = np.array([data_s, data_p, data_i])
data.shape = ((-1, Ny, Nx))
data_spect.shape = ((Nt, Nx))

spect = data_spect[::-1, ...]
spect_int = data_spect_int[::-1]

##############
fig_int_spect, ax = plt.subplots(1, 1, figsize=(5, 5))
for idx in range(3):
    ax.scatter(omega[idx_spi[idx]] * Omega_R * 10 ** 3,
            spect_int[idx_spi[idx]], c=color_spi[idx],
            marker=u'o', s=100)
ax.plot(omega * Omega_R * 10 ** 3, spect_int, 'black')
ax.set_xlim(-3, +3)
ax.set_xlim(ax.get_xlim()[::-1])
ax.xaxis.set_ticks(np.arange(-2, 3, 2))
ax.set_ylim(0, 9000)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.close(fig_int_spect)
#fig_int_spect.close()
#############

side_ky = np.pi / a_y
side_kx = np.pi / a_x
ky, delta_ky = np.linspace(-(2 * (Ny - 1) / float(Ny) - 1) * side_ky, side_ky, num=Ny, retstep=True)
kx, delta_kx = np.linspace(-(2 * (Nx - 1) / float(Nx) - 1) * side_kx, side_kx, num=Nx, retstep=True)
KX, KY = np.meshgrid(kx, ky)
kx_label = r'$k_x[\mu m^{-1}]$'

LP = enLP(KX, KY) * 10 ** 3
UP = enUP(KX, KY) * 10 ** 3
G = gamma(KX, KY) * 10 ** 3 / 2

############
fig_spect, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.clip(np.log10(spect), -6, 1), cmap=cm.binary, origin='lower',
          extent=[kx[0] / lC, kx[-1] / lC,
              omega[0] * Omega_R * 10 ** 3, omega[-1] * Omega_R * 10 ** 3])

for idx in range(3):
    ax.scatter(momentum_spi[idx, 0] / lC, energy_spi[idx] * Omega_R * 10 ** 3, c=color_spi[idx], marker=u'o', s=60)

ax.plot(KX[Ny / 2, :] / lC, UP[Ny / 2, :] - omega_X * 10 ** 3, 'k-.')

ax.fill_between(KX[Ny / 2, :] / lC, LP[Ny / 2, :] - G[Ny / 2, :] / 2 - omega_X * 10 ** 3,
                LP[Ny / 2, :] + G[Ny / 2, :] / 2 - omega_X * 10 ** 3, color='0.', alpha=0.5)

ax.axhline(y=enLP(k_pmpx, k_pmpy) * 10 ** 3 + np.sqrt(3) * gp * 10 ** 3 / 2 - omega_X * 10 ** 3, color='r', ls='dashed')

ax.axis('image')
ax.set_ylim(-3, +3)
ax.yaxis.set_ticks(np.arange(-2, 3, 2))
ax.set_ylabel(r'$\omega - \omega_X$ [meV]')
ax.set_xlim(kx[20] / lC, kx[-20] / lC)
ax.set_xlabel(kx_label)
plt.close(fig_spect)
#fig_spect.close()
##################

def_x_pos = 11.
def_y_pos = -0.546875
idx_def_x_pos = find_index(def_x_pos, Lx, a_x)
idx_def_y_pos = find_index(def_y_pos, Ly, a_y)
x_def = x[idx_def_x_pos]
y_def = y[idx_def_y_pos]

#checks
en_LPB = enLP(0, 0)
en_UPB = enUP(0, 0)
print 2 * momentum_spi[1] - momentum_spi[0] - momentum_spi[2]
print en_UPB - en_LPB - 2 * Omega_R
print enLP(k_sigx, k_sigy) + enLP(2 * k_pmpx - k_sigx, 2 * k_pmpy - k_sigy) - 2 * enLP(k_pmpx, k_pmpy)
print 2 * energy_spi[1] - energy_spi[0] - energy_spi[2]

##################
fig_data, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(data[idx, ...],
                     cmap=cm.gray, origin='lower', extent=lC * np.array([x[0], x[-1], y[0], y[-1]]))
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
plt.close(fig_data)
#fig_data.close()
##################

##################
fig_data_slice, axes = plt.subplots(2, 3, figsize=(12, 12))

for idx in range(3):
    axes[0, idx].set_title(titles[idx])
    axes[0, idx].plot(lC * x, data[idx, idx_def_y_pos, :])
    axes[0, idx].grid()
    axes[1, idx].plot(lC * y, data[idx, :, idx_def_x_pos])
    axes[1, idx].grid()

for ax in range(3):
    axes[0, ax].set_xlabel(x_label)
    axes[1, ax].set_xlabel(y_label)
    axes[0, ax].set_xlim((x_def - 10) * lC, (x_def + 10) * lC)
    axes[1, ax].set_xlim((y_def - 10) * lC, (y_def + 10) * lC)
plt.close(fig_data_slice)
#fig_data_slice.close()
##################

#################
#################
grad_s = np.gradient(data[0, ...])
grad_p = np.gradient(data[1, ...])
grad_i = np.gradient(data[2, ...])
##################
fig_grad, axes = plt.subplots(2, 3, figsize=(12, 12))

axes[0, 0].set_title('gradient of signal, x component')
axes[0, 0].imshow(grad_s[1], cmap=cm.gray, origin='lower', extent=lC * np.array([x[0], x[-1], y[0], y[-1]]))
axes[1, 0].plot(lC * x, grad_s[1][idx_def_y_pos, :])
axes[1, 0].grid()

axes[0, 0].set_ylabel(y_label)
axes[1, 0].set_xlabel(x_label)
axes[1, 0].set_xlim((x_def - 10) * lC, (x_def + 10) * lC)


axes[0, 1].set_title('gradient of pump, x component')
axes[0, 1].imshow(grad_p[1], cmap=cm.gray, origin='lower', extent=lC * np.array([x[0], x[-1], y[0], y[-1]]))
axes[1, 1].plot(lC * x, grad_p[1][idx_def_y_pos, :])
axes[1, 1].grid()

axes[1, 1].set_xlabel(x_label)
axes[1, 1].set_xlim((x_def - 10) * lC, (x_def + 10) * lC)


axes[0, 2].set_title('gradient of idler, x component')
axes[0, 2].imshow(grad_i[1], cmap=cm.gray, origin='lower', extent=lC * np.array([x[0], x[-1], y[0], y[-1]]))
axes[1, 2].plot(lC * x, grad_i[1][idx_def_y_pos, :])
axes[1, 2].grid()

axes[1, 2].set_xlabel(x_label)
axes[1, 2].set_xlim((x_def - 10) * lC, (x_def + 10) * lC)
plt.close(fig_grad)
#fig_grad.close()
##################


print 'drag of signal', grad_s[1][idx_def_y_pos, idx_def_x_pos]
print 'drag of pump', grad_p[1][idx_def_y_pos, idx_def_x_pos]
print 'drag of idler', grad_i[1][idx_def_y_pos, idx_def_x_pos]
