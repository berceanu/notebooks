import numpy as np
import numpy.linalg as LA
import scipy.ndimage as scimg
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ipdb
from matplotlib.patches import Rectangle
#import string

#ipdb.set_trace()

lC = 0.868

Omega_R = 4.4 / 2 * 10 ** (-3)
omega_C0 = omega_X = 1.53

energy_spi = np.array([-0.748, -0.2, 0.349])
print 2 * energy_spi[1] - energy_spi[0] - energy_spi[2]

k_sigx = - 0.179
k_pmpx = 1.391
k_idlx = 2.962

k_sigy = 0.
k_pmpy = 0.
k_idly = 0.

momentum_spi = np.array([[k_sigx, k_sigy],
                         [k_pmpx, k_pmpy],
                         [k_idlx, k_idly]])
print 2 * momentum_spi[1] - momentum_spi[0] - momentum_spi[2]

kz = 1 / lC * np.sqrt(omega_C0 / (2 * Omega_R))

def enC(kx, ky):
    return omega_C0 * np.sqrt(1 + (np.sqrt(kx ** 2 + ky ** 2) / kz) ** 2)

def enLP(kx, ky):
    return (enC(kx, ky) + omega_X) / 2 - np.sqrt(Omega_R ** 2 + ((enC(kx, ky) - omega_X) / 2) ** 2)

def enUP(kx, ky):
    return (enC(kx, ky) + omega_X) / 2 + np.sqrt(Omega_R ** 2 + ((enC(kx, ky) - omega_X) / 2) ** 2)

en_LPB = enLP(0, 0)
en_UPB = enUP(0, 0)
print en_UPB - en_LPB - 2 * Omega_R

def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + (Omega_R / (enLP(kx, ky) - enC(kx, ky))) ** 2)

gamma_X = 2 * 0.12 * Omega_R
gamma_C = 2 * 0.12 * Omega_R

def gamma(kx, ky):
    return gamma_C + hopf_x(kx, ky) ** 2 * (gamma_X - gamma_C)

gp = gamma(k_pmpx, k_pmpy)

print enLP(k_sigx, k_sigy) + enLP(2 * k_pmpx - k_sigx, 2 * k_pmpy - k_sigy) - 2 * enLP(k_pmpx, k_pmpy)

mpl.rcParams.update({'font.size': 22, 'font.family': 'serif'})

phi_golden = 1.618  # golden ratio
x_label = r'$x[\mu m]$'
y_label = r'$y[\mu m]$'
cut_label = r'$r [\mu m]$'

kx_label = r'$k_x[\mu m^{-1}]$'
ky_label = r'$k_y[\mu m^{-1}]$'

titles = ['signal', 'pump', 'idler']
color_spi = ['blue', 'red', 'green']

def find_index(dist_or_mom, side, delta):
    return int(np.around((side + dist_or_mom) / delta))

def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha / 2
    w[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha / 2)
    w[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2)))

    return w


def calc_distance(a, b):
    idx_y0 = find_index(a[1], Ly, a_y)
    idx_x0 = find_index(a[0], Lx, a_x)
    v1 = np.array([idx_y0, idx_x0])

    idx_y1 = find_index(b[1], Ly, a_y)
    idx_x1 = find_index(b[0], Lx, a_x)
    v2 = np.array([idx_y1, idx_x1])

    dst = int(np.ceil(LA.norm(v2 - v1))) + 1

    t = np.linspace(0, 1, dst)
    v = v2[..., None] * t + v1[..., None] * (1 - t)

    return (dst, v)


def fft_fit(cut):
    cut_fft = np.fft.rfft(cut, n_points)
    cut_ampl_spect = np.abs(cut_fft) * (2. / n_cut)
    cut_phase_spect = np.angle(cut_fft)

    idx_k = np.argmax(cut_ampl_spect)
    wave_amplitude = np.amax(cut_ampl_spect)
    k = k_cut[idx_k]
    phi = cut_phase_spect[idx_k]

    return (wave_amplitude, k, cut_ampl_spect, phi)

Ny = 256
Ly = 70.
y, a_y = np.linspace(-Ly, Ly, num=Ny, endpoint=False, retstep=True)

side_ky = np.pi / a_y
ky, delta_ky = np.linspace(-(2 * (Ny - 1) / float(Ny) - 1) * side_ky, side_ky, num=Ny, retstep=True)


Nx = 256
Lx = 70.
x, a_x = np.linspace(-Lx, Lx, num=Nx, endpoint=False, retstep=True)

side_kx = np.pi / a_x
kx, delta_kx = np.linspace(-(2 * (Nx - 1) / float(Nx) - 1) * side_kx, side_kx, num=Nx, retstep=True)


KX, KY = np.meshgrid(kx, ky)


Nt = 2048
dxsav_sp = 1.
t, delta_t = np.linspace(0, Nt - 1, num=Nt, endpoint=False, retstep=True)

side_omega = dxsav_sp * np.pi / delta_t
omega, delta_omega = np.linspace(-side_omega, side_omega, num=Nt, endpoint=False, retstep=True)


idx_spi = np.zeros((3,), dtype=np.int)
for idx in range(3):
    idx_spi[idx] = find_index(energy_spi[idx], side_omega, delta_omega)


def find_index(dist_or_mom, side, delta):
    return int(np.around((side + dist_or_mom) / delta))


base = '/home/berceanu/ownCloud/Dropbox/'
folder = r'GP_data/OPO/tophat/defect/kp/1.4/fp/0.11/N/256/move_defect/spectrum_2048/'

path_s = base + folder + r'opo_ph-spc_enfilt_signal.dat'
path_p = base + folder + r'opo_ph-spc_enfilt_pump.dat'
path_i = base + folder + r'opo_ph-spc_enfilt_idler.dat'

path_s_mom = base + folder + r'opo_ph-mom_enfilt_signal.dat'
path_p_mom = base + folder + r'opo_ph-mom_enfilt_pump.dat'
path_i_mom = base + folder + r'opo_ph-mom_enfilt_idler.dat'

path_spect = base + folder + r'spectr_om-vs-kx_no-trigg.dat'
path_spect_int = base + folder + r'int-spectr_no-trigg.dat'

data_s = np.loadtxt(path_s, usecols=(2,))
data_p = np.loadtxt(path_p, usecols=(2,))
data_i = np.loadtxt(path_i, usecols=(2,))

data_s_mom = np.loadtxt(path_s_mom, usecols=(2,))
data_p_mom = np.loadtxt(path_p_mom, usecols=(2,))
data_i_mom = np.loadtxt(path_i_mom, usecols=(2,))

data_spect = np.loadtxt(path_spect, usecols=(2,))
data_spect_int = np.loadtxt(path_spect_int, usecols=(1,))

data = np.array([data_s, data_p, data_i])
data_mom = np.array([data_s_mom, data_p_mom, data_i_mom])
data.shape = ((-1, Ny, Nx))
data_mom.shape = ((-1, Ny, Nx))
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
#ax.set_ylabel(r'$\sum_k\left|\psi_C(k,\omega)\right|^2$')
#ax.set_xlabel(r'$\omega - \omega_X$ [meV]')
fig_int_spect.savefig('fig_GP_int_spect', bbox_inches='tight')
#############

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
#ax.set_title(r'$\left|\psi_C(k,\omega)\right|^2$')
ax.set_ylim(-3, +3)
ax.yaxis.set_ticks(np.arange(-2, 3, 2))
ax.set_ylabel(r'$\omega - \omega_X$ [meV]')
ax.set_xlim(kx[20] / lC, kx[-20] / lC)
ax.set_xlabel(kx_label)
fig_spect.savefig('fig_GP_spect', bbox_inches='tight')
##################

fig_data_init, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(data[idx, ...],
                     cmap=cm.gray, origin='lower', extent=lC * np.array([x[0], x[-1], y[0], y[-1]]))
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
fig_data_init.savefig('fig_GP_data_init', bbox_inches='tight')

def_x_pos = 11.
def_y_pos = -0.546875

idx_def_x_pos = find_index(def_x_pos, Lx, a_x)
idx_def_y_pos = find_index(def_y_pos, Ly, a_y)

x_def = x[idx_def_x_pos]
y_def = y[idx_def_y_pos]


side = 15. / lC  # half-side of square centered on defect
side_in = 7. / lC

xl = x_def - side
xr = x_def + side
yb = y_def - side
yt = y_def + side

xl_in = x_def - side_in
xr_in = x_def + side_in
yb_in = y_def - side_in
yt_in = y_def + side_in

idx_xl = find_index(xl, Lx, a_x)
idx_xr = find_index(xr, Lx, a_x)
idx_yb = find_index(yb, Ly, a_y)
idx_yt = find_index(yt, Ly, a_y)

idx_xl_in = find_index(xl_in, Lx, a_x)
idx_xr_in = find_index(xr_in, Lx, a_x)
idx_yb_in = find_index(yb_in, Ly, a_y)
idx_yt_in = find_index(yt_in, Ly, a_y)

###
fig_data_cut_s, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(data[0, idx_yb:idx_yt, idx_xl:idx_xr],
                 cmap=cm.gray, origin='lower',
                 extent=lC * np.array([x[idx_xl], x[idx_xr],
                     y[idx_yb], y[idx_yt]]))
#ax.scatter(lC * x_def, lC * y_def, s=50, c=u'orange', marker=u'o')
#ax.axis('image')
#ax.set_title(titles[0])
ax.set_ylabel(y_label)
ax.yaxis.set_ticks(np.arange(-10, 11, 5))
#ax.set_xlabel(x_label)
ax.set_xticklabels([])
ax.set_xlim(lC * x[idx_xl], lC * x[idx_xr])
ax.set_ylim(lC * y[idx_yb], lC * y[idx_yt])
# rectangle with lower left at (x, y)
ax.add_patch(Rectangle((x[idx_xl_in], y[idx_yb_in]),
    x[idx_xr_in] - x[idx_xl_in],
    y[idx_yt_in] - y[idx_yb_in], fill=False))
fig_data_cut_s.savefig('fig_GP_data_cut_s', bbox_inches='tight')
####

###
fig_data_cut_p, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.imshow(data[1, idx_yb:idx_yt, idx_xl:idx_xr],
                 cmap=cm.gray, origin='lower', extent=lC * np.array([x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]]))
#ax.scatter(lC * x_def, lC * y_def, s=50, c=u'orange', marker=u'o')
ax.axis('image')
#ax.set_title(titles[1])

ax.set_ylabel(y_label)
ax.yaxis.set_ticks(np.arange(-10, 11, 5))
#ax.set_xlabel(x_label)
ax.set_xticklabels([])
ax.set_xlim(lC * x[idx_xl], lC * x[idx_xr])
ax.set_ylim(lC * y[idx_yb], lC * y[idx_yt])
fig_data_cut_p.savefig('fig_GP_data_cut_p', bbox_inches='tight')
####

###
fig_data_cut_i, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.imshow(data[2, idx_yb:idx_yt, idx_xl:idx_xr],
                 cmap=cm.gray, origin='lower', extent=lC * np.array([x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]]))
#ax.scatter(lC * x_def, lC * y_def, s=50, c=u'orange', marker=u'o')
ax.axis('image')
#ax.set_title(titles[2])

ax.set_ylabel(y_label)
ax.yaxis.set_ticks(np.arange(-10, 11, 5))
ax.set_xlabel(x_label)
#ax.set_xticklabels([])
ax.set_xlim(lC * x[idx_xl], lC * x[idx_xr])
ax.set_ylim(lC * y[idx_yb], lC * y[idx_yt])
fig_data_cut_i.savefig('fig_GP_data_cut_i', bbox_inches='tight')
####

my_ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=a_y))
my_kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=a_x))


data_fft = np.fft.fftshift(np.fft.fft2(data), axes=(1, 2))  # removed np.real


fig_data_fft, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(np.log10(np.abs(data_fft[idx, :, :])),
                     cmap=cm.gray, origin='lower', extent=np.array([my_kx[0], my_kx[-1], my_ky[0], my_ky[-1]]) / lC)
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(ky_label)
for ax in range(3):
    axes[ax].set_xlabel(kx_label)
fig_data_fft.savefig('fig_GP_data_fft', bbox_inches='tight')

kyb = kxl = - 2. * lC
kyt = kxr = 2. * lC

idx_kyb = find_index(kyb, np.abs(my_ky[0]), my_ky[1] - my_ky[0])
idx_kyt = find_index(kyt, np.abs(my_ky[0]), my_ky[1] - my_ky[0])
idx_kxl = find_index(kxl, np.abs(my_kx[0]), my_kx[1] - my_kx[0])
idx_kxr = find_index(kxr, np.abs(my_kx[0]), my_kx[1] - my_kx[0])


fig_data_fft_cut, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(np.log10(np.abs(data_fft[idx, idx_kyb:idx_kyt, idx_kxl:idx_kxr])),
                     cmap=cm.gray, origin='lower', extent=np.array([my_kx[idx_kxl], my_kx[idx_kxr], my_ky[idx_kyb], my_ky[idx_kyt]]) / lC)
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(ky_label)
for ax in range(3):
    axes[ax].set_xlabel(kx_label)
fig_data_fft_cut.savefig('fig_GP_data_fft_cut', bbox_inches='tight')

il = ib = 72
ir = it = 184

tukey_length = data_fft[1, ib:it, il:ir].shape[-1]
tukey_alpha = 0.6
tw_x = tw_y = tukeywin(tukey_length, tukey_alpha)
tw_2d = np.outer(tw_y, tw_x)


fig_tukey, ax = plt.subplots(1, 1, figsize=(phi_golden * 4, 4))

ax.plot(my_kx[il:ir] / lC, tw_x, '-^')

ax.set_xlabel(kx_label)
ax.set_xlim(0., my_kx[ir] / lC)
ax.set_ylim(0., 1.1)
ax.set_title('tukey profile')
fig_tukey.savefig('fig_GP_tukey', bbox_inches='tight')

mask = np.zeros((Ny, Nx))
mask[ib:it, il:ir] = tw_2d

low_pass = data_fft * mask
high_pass = data_fft - low_pass


low_pass_spc = np.real(np.fft.ifft2(np.fft.ifftshift(low_pass, axes=(1, 2))))
high_pass_spc = np.real(np.fft.ifft2(np.fft.ifftshift(high_pass, axes=(1, 2))))


fig_low_pass, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(low_pass_spc[idx, idx_yb:idx_yt, idx_xl:idx_xr],
                     cmap=cm.gray, origin='lower', extent=np.array([x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]]) * lC)
    axes[idx].scatter(x_def * lC, y_def * lC, s=50, c=u'orange', marker=u'o')
    axes[idx].axis('image')
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
    axes[ax].set_xlim(lC * x[idx_xl], lC * x[idx_xr])
    axes[ax].set_ylim(lC * y[idx_yb], lC * y[idx_yt])
fig_low_pass.savefig('fig_GP_low_pass', bbox_inches='tight')

###
fig_high_pass_s, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(high_pass_spc[0, idx_yb_in:idx_yt_in, idx_xl_in:idx_xr_in],
                 cmap=cm.binary, origin='lower',
                 extent=np.array([x[idx_xl_in], x[idx_xr_in],
                     y[idx_yb_in], y[idx_yt_in]]) * lC)
#ax.scatter(x_def * lC, y_def * lC, s=50, c=u'orange', marker=u'o')
#ax.axis('image')
#ax.set_title(titles[0])
#ax.set_ylabel(y_label)
#ax.set_xlabel(x_label)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(lC * x[idx_xl_in], lC * x[idx_xr_in])
ax.set_ylim(lC * y[idx_yb_in], lC * y[idx_yt_in])
fig_high_pass_s.savefig('fig_GP_high_pass_s', bbox_inches='tight')
###

angle = np.pi
length = 27 * np.sqrt((a_x ** 2 + a_y ** 2) / 2)
x1 = length * np.cos(angle) + x_def * lC
y1 = length * np.sin(angle) + y_def * lC
back_angle = -np.pi + angle
back_length = length - 6 * np.sqrt((a_x ** 2 + a_y ** 2) / 2)
x0 = back_length * np.cos(back_angle) + x1
y0 = back_length * np.sin(back_angle) + y1


side_cut = LA.norm([x1 - x0, y1 - y0])


n_cut, positions = calc_distance((x0, y0), (x1, y1))


x_cut = np.linspace(0, side_cut, n_cut)
delta_cut = side_cut / n_cut


n_points = n_cut * 5

k_cut = 2 * np.pi * np.fft.rfftfreq(n_points, d=delta_cut)


ampl = np.zeros(3)
mom = np.zeros(3)
phase = np.zeros(3)

wave = np.zeros((3, n_cut))
wave_fft = np.zeros((3, n_points // 2 + 1))


for idx in range(3):
    wave[idx, :] = scimg.interpolation.map_coordinates(high_pass_spc[idx], positions)
    ampl[idx], mom[idx], wave_fft[idx, :], phase[idx] = fft_fit(wave[idx, :])


fig_high_pass_fit, axes = plt.subplots(3, 3, figsize=(16, 14))

for col in range(3):
    axes[0, col].imshow(high_pass_spc[col, idx_yb:idx_yt, idx_xl:idx_xr],
                        cmap=cm.gray, origin='lower', extent=np.array([x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]]) * lC)
    axes[0, col].plot([x0, x1], [y0, y1], 'o-', color=u'orange')
    axes[0, col].plot([x_def * lC, x0], [y_def * lC, y0], 'o--', color=u'orange')
    axes[0, col].axis('image')
    axes[0, col].set_title(titles[col])
    axes[0, col].set_xlabel(x_label)
    axes[0, col].set_xlim(lC * x[idx_xl], lC * x[idx_xr])
    axes[0, col].set_ylim(lC * y[idx_yb], lC * y[idx_yt])

    axes[1, col].plot(k_cut / lC, wave_fft[col], '--o')
    axes[1, col].set_xlabel(r'$k [\mu m^{-1}]$')

    axes[2, col].plot(x_cut * lC, wave[col, :], '-')
    axes[2, col].plot(x_cut * lC, ampl[col] * np.cos(mom[col] * x_cut + phase[col]), '--')
    axes[2, col].set_xlabel(cut_label)

axes[0, 0].set_ylabel(y_label)
axes[1, 0].set_ylabel('magnitude')
fig_high_pass_fit.savefig('fig_GP_high_pass_fit', bbox_inches='tight')

print mom / lC  # extracted momenta, in mu^(-1)

dot_coords = np.transpose(np.vstack((mom * np.cos(angle), mom * np.sin(angle))))
shifted_dot_coords = momentum_spi + dot_coords

fig_data_mom_s, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.clip(np.log10(data_mom[0, ...]), -7, 1),
                 cmap=cm.gray, origin='lower',
                 extent=np.array([kx[0], kx[-1], ky[0], ky[-1]]) / lC)
#ax.set_title(titles[0])

ax.scatter(momentum_spi[0, 0] / lC, momentum_spi[0, 1] / lC, c=color_spi[0], marker=u'o', s=100)
ax.scatter(shifted_dot_coords[0, 0] / lC, shifted_dot_coords[0, 1] / lC, c=color_spi[0], marker=u'o', s=100)

ax.set_ylabel(ky_label)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.xaxis.set_ticks(np.arange(-4, 5, 2))
ax.yaxis.set_ticks(np.arange(-4, 5, 2))
#ax.set_xlabel(kx_label)
ax.set_xticklabels([])
#ax.set_xlim(momentum_spi[0, 0] / lC - 3.5, momentum_spi[0, 0] / lC + 3.5)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
fig_data_mom_s.savefig('fig_GP_data_mom_s', bbox_inches='tight')


fig_data_mom_p, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.clip(np.log10(data_mom[1, ...]), -7, 1),
                 cmap=cm.gray, origin='lower',
                 extent=np.array([kx[0], kx[-1], ky[0], ky[-1]]) / lC)
#ax.set_title(titles[0])

ax.scatter(momentum_spi[1, 0] / lC, momentum_spi[1, 1] / lC, c=color_spi[1], marker=u'o', s=100)
ax.scatter(shifted_dot_coords[1, 0] / lC, shifted_dot_coords[1, 1] / lC, c=color_spi[1], marker=u'o', s=100)

ax.set_ylabel(ky_label)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.xaxis.set_ticks(np.arange(-4, 5, 2))
ax.yaxis.set_ticks(np.arange(-4, 5, 2))
#ax.set_xlabel(kx_label)
ax.set_xticklabels([])
#ax.set_xlim(momentum_spi[1, 0] / lC - 3.5, momentum_spi[1, 0] / lC + 3.5)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
fig_data_mom_p.savefig('fig_GP_data_mom_p', bbox_inches='tight')


fig_data_mom_i, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.clip(np.log10(data_mom[2, ...]), -7, 1),
                 cmap=cm.gray, origin='lower',
                 extent=np.array([kx[0], kx[-1], ky[0], ky[-1]]) / lC)
#ax.set_title(titles[0])

ax.scatter(momentum_spi[2, 0] / lC, momentum_spi[2, 1] / lC, c=color_spi[2], marker=u'o', s=100)
ax.scatter(shifted_dot_coords[2, 0] / lC, shifted_dot_coords[2, 1] / lC, c=color_spi[2], marker=u'o', s=100)

ax.set_ylabel(ky_label)
ax.set_xlabel(kx_label)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.xaxis.set_ticks(np.arange(-4, 5, 2))
ax.yaxis.set_ticks(np.arange(-4, 5, 2))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
#ax.set_xlim(momentum_spi[2, 0] / lC - 3.5, momentum_spi[2, 0] / lC + 3.)
fig_data_mom_i.savefig('fig_GP_data_mom_i', bbox_inches='tight')
