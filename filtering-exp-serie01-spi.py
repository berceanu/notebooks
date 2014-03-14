import numpy as np
import numpy.linalg as LA
import scipy.ndimage as scimg
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


kz = 20.0
Omega_R = 6 / 2. * 10 ** (-3)
en_LPB = 1481 * 10 ** (-3)
omega_X = 1484 * 10 ** (-3)
en_UPB = 1487 * 10 ** (-3)


print en_UPB - en_LPB - 2 * Omega_R


omega_C0 = omega_X + 2 * np.sqrt(((en_UPB - en_LPB) / 2) ** 2 - Omega_R ** 2)
print omega_C0 - omega_X


energy_spi = 2 * np.pi * 29.979 * 6.582 / np.array([836.62, 836.28, 835.94])


k_sigx = 0.41
k_pmpx = 0.90
k_idlx = 1.39

k_sigy = 0.
k_pmpy = 0.
k_idly = 0.

momentum_spi = np.array([[k_sigx, k_sigy],
                         [k_pmpx, k_pmpy],
                         [k_idlx, k_idly]])
print 2 * momentum_spi[1] - momentum_spi[0] - momentum_spi[2]  # 0


gX = 5 * 10 ** (-3) * 10 ** (-3)


ns_chosen = 0.4 * gX
np_chosen = 650 * gX
ni_chosen = 0.01 * gX



def enC(kx, ky):
    return omega_C0 * np.sqrt(1 + (np.sqrt(kx ** 2 + ky ** 2) / kz) ** 2)


def enLP(kx, ky):
    return (enC(kx, ky) + omega_X) / 2 - np.sqrt(Omega_R ** 2 + ((enC(kx, ky) - omega_X) / 2) ** 2)


def enUP(kx, ky):
    return (enC(kx, ky) + omega_X) / 2 + np.sqrt(Omega_R ** 2 + ((enC(kx, ky) - omega_X) / 2) ** 2)


def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + (Omega_R / (enLP(kx, ky) - enC(kx, ky))) ** 2)


b_shift = 0.355 * 10 ** (-3)


def blue_enLP(kx, ky):
    # (ni_chosen + np_chosen + ns_chosen)
    return enLP(kx, ky) + 2 * hopf_x(kx, ky) ** 2 * b_shift



gamma_X = 2. * 10 ** (-6)
gamma_C = 0.2 * 10 ** (-3)


def gamma(kx, ky):
    return gamma_C + hopf_x(kx, ky) ** 2 * (gamma_X - gamma_C)


gp = gamma(k_pmpx, k_pmpy)


hbar = 0.00413566751 / (2 * np.pi)


mpl.rcParams.update({'font.size': 16, 'font.family': 'serif'})


x_label = r'$x[\mu m]$'
y_label = r'$y[\mu m]$'
cut_label = r'$r [\mu m]$'

kx_label = r'$k_x[\mu m^{-1}]$'
ky_label = r'$k_y[\mu m^{-1}]$'

titles = ['signal', 'pump', 'idler']
color_spi = ['blue', 'red', 'green']


nk = 801
kl = -4.
kr = 4.
kx = np.linspace(kl, kr, num=nk)
ky = np.linspace(kl, kr, num=nk)
KX, KY = np.meshgrid(kx, ky)


LP = enLP(KX, KY)
UP = enUP(KX, KY)
BLP = blue_enLP(KX, KY)
G = gamma(KX, KY) / 2


fig_disp, ax = plt.subplots()

ax.plot(KX[nk / 2, :], LP[nk / 2, :], 'k-.')
ax.plot(KX[nk / 2, :], UP[nk / 2, :], 'k-.')

ax.fill_between(KX[nk / 2, :], BLP[nk / 2, :] - G[nk / 2, :] / 2,
                BLP[nk / 2, :] + G[nk / 2, :] / 2, color='0.75', alpha=0.5)

ax.axhline(y=enLP(k_pmpx, k_pmpy) + np.sqrt(3)
           * gp / 2, color='r', ls='dashed')

for idx in range(3):
    ax.scatter(momentum_spi[idx, 0], energy_spi[idx], c=color_spi[idx])

ax.set_title('dispersion')
ax.set_xlim(kl, kr)
ax.set_ylim(enLP(0, 0), 1.49)
ax.set_xlabel(kx_label)
ax.set_ylabel(r'$\epsilon[eV]$')
fig_disp.savefig('fig_exp_disp', bbox_inches='tight')

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
    w[first_condition] = 0.5 * \
        (1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha / 2)
    w[third_condition] = 0.5 * \
        (1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2)))

    return w


def calc_distance(a, b):
    idx_y0 = find_index(a[1], side_y, delta_y)
    idx_x0 = find_index(a[0], side_x, delta_x)
    v1 = np.array([idx_y0, idx_x0])

    idx_y1 = find_index(b[1], side_y, delta_y)
    idx_x1 = find_index(b[0], side_x, delta_x)
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


ny = 250
delta_y = 0.64
side_y = ny * delta_y / 2
y = np.arange(-side_y, side_y, delta_y)


nx = 1024
delta_x = 0.16
side_x = nx * delta_x / 2
x = np.arange(-side_x, side_x, delta_x)


base = '/home/berceanu/ownCloud/Dropbox/experiments_OPO-defect/'
folder = 'OPO_exp_2012_aug_29_data+fig_new/'

path_s = base + folder + 'an_OPO-serie01-signal-original_590_04.dat'
path_p = base + folder + 'an_OPO-serie01-pump-original_636_04.dat'
path_i = base + folder + 'an_OPO-serie01-idler-original_682_04.dat'

data_s = np.loadtxt(path_s)
data_p = np.loadtxt(path_p)
data_i = np.loadtxt(path_i)

data = np.array([data_s, data_p, data_i])


fig_data_init, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(data[idx, :, :],
                     cmap=cm.binary, origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
fig_data_init.savefig('fig_exp_data_init', bbox_inches='tight')

x_def = 14.
y_def = -5.

side = 35


xl = x_def - side
xr = x_def + side
yb = y_def - side
yt = y_def + side

idx_xl = find_index(xl, side_x, delta_x)
idx_xr = find_index(xr, side_x, delta_x)
idx_yb = find_index(yb, side_y, delta_y)
idx_yt = find_index(yt, side_y, delta_y)


fig_data_cut, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(data[idx, idx_yb:idx_yt, idx_xl:idx_xr],
                     cmap=cm.binary, origin='lower', extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
    axes[idx].scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
    axes[idx].axis('image')
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
    axes[ax].set_xlim(x[idx_xl], x[idx_xr])
    axes[ax].set_ylim(y[idx_yb], y[idx_yt])
fig_data_cut.savefig('fig_exp_data_cut', bbox_inches='tight')

ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=delta_y))
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=delta_x))


data_fft = np.fft.fftshift(np.fft.fft2(data), axes=(1, 2))  # removed np.real


fig_data_fft, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx in range(3):
    axes[idx].imshow(np.log10(np.abs(data_fft[idx, :, :])),
                     cmap=cm.binary, origin='lower', extent=[kx[0], kx[-1], ky[0], ky[-1]])
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(ky_label)
for ax in range(3):
    axes[ax].set_xlabel(kx_label)
fig_data_fft.savefig('fig_exp_data_fft', bbox_inches='tight')

kyb = kxl = -3
kyt = kxr = 3

idx_kyb = find_index(kyb, np.abs(ky[0]), ky[1] - ky[0])
idx_kyt = find_index(kyt, np.abs(ky[0]), ky[1] - ky[0])
idx_kxl = find_index(kxl, np.abs(kx[0]), kx[1] - kx[0])
idx_kxr = find_index(kxr, np.abs(kx[0]), kx[1] - kx[0])


fig_data_fft_cut, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(
        np.log10(np.abs(data_fft[idx, idx_kyb:idx_kyt, idx_kxl:idx_kxr])),
        cmap=cm.binary, origin='lower', extent=[kx[idx_kxl], kx[idx_kxr], ky[idx_kyb], ky[idx_kyt]])
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(ky_label)
for ax in range(3):
    axes[ax].set_xlabel(kx_label)
fig_data_fft_cut.savefig('fig_exp_data_fft_cut', bbox_inches='tight')

ib = 115
it = 137
il = 501
ir = 523


tukey_length = data_fft[1, ib:it, il:ir].shape[-1]
tukey_alpha = 0.3
tw_x = tw_y = tukeywin(tukey_length, tukey_alpha)
tw_2d = np.outer(tw_y, tw_x)


fig_tukey, ax = plt.subplots(1, 1, figsize=(6, 2))

ax.plot(kx[il:ir], tw_x, '-^')

ax.set_xlabel(kx_label)
ax.set_xlim(0., kx[ir])
ax.set_ylim(0., 1.1)
ax.set_title('tukey profile')
fig_tukey.savefig('fig_exp_tukey', bbox_inches='tight')

mask = np.zeros((ny, nx))
mask[ib:it, il:ir] = tw_2d


low_pass = data_fft * mask
high_pass = data_fft - low_pass


low_pass_spc = np.real(np.fft.ifft2(np.fft.ifftshift(low_pass, axes=(1, 2))))
high_pass_spc = np.real(np.fft.ifft2(np.fft.ifftshift(high_pass, axes=(1, 2))))


fig_low_pass, axes = plt.subplots(1, 3, figsize=(12, 12))

for idx in range(3):
    axes[idx].imshow(low_pass_spc[idx, idx_yb:idx_yt, idx_xl:idx_xr],
                     cmap=cm.binary, origin='lower', extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
    axes[idx].scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
    axes[idx].axis('image')
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
    axes[ax].set_xlim(x[idx_xl], x[idx_xr])
    axes[ax].set_ylim(y[idx_yb], y[idx_yt])
fig_low_pass.savefig('fig_exp_low_pass', bbox_inches='tight')


fig_high_pass, axes = plt.subplots(1, 3, figsize=(18, 18))

for idx in range(3):
    axes[idx].imshow(high_pass_spc[idx, idx_yb:idx_yt, idx_xl:idx_xr],
                     cmap=cm.binary, origin='lower', extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
    axes[idx].scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
    axes[idx].axis('image')
    axes[idx].set_title(titles[idx])

axes[0].set_ylabel(y_label)
for ax in range(3):
    axes[ax].set_xlabel(x_label)
    axes[ax].set_xlim(x[idx_xl], x[idx_xr])
    axes[ax].set_ylim(y[idx_yb], y[idx_yt])
fig_high_pass.savefig('fig_exp_high_pass', bbox_inches='tight')

angle = -12 * np.pi / 180
length = 65 * np.sqrt((delta_x ** 2 + delta_y ** 2) / 2)
x1 = length * np.cos(angle) + x_def
y1 = length * np.sin(angle) + y_def
back_angle = -np.pi + angle
back_length = length - 15 * np.sqrt((delta_x ** 2 + delta_y ** 2) / 2)
x0 = back_length * np.cos(back_angle) + x1
y0 = back_length * np.sin(back_angle) + y1


side_cut = LA.norm([x1 - x0, y1 - y0])


n_cut, positions = calc_distance((x0, y0), (x1, y1))


x_cut = np.linspace(0, side_cut, n_cut)
delta_cut = side_cut / n_cut


n_points = 400


k_cut = 2 * np.pi * np.fft.rfftfreq(n_points, d=delta_cut)
idx_cut_4 = int(np.around(4. / np.diff(k_cut)[0]))


ampl = np.zeros(3)
mom = np.zeros(3)
phase = np.zeros(3)

wave = np.zeros((3, n_cut))
wave_fft = np.zeros((3, n_points // 2 + 1))


def sliceImage(I, pos_vect, *arg, **kws):
    return


for idx in range(3):
    wave[idx, :] = scimg.interpolation.map_coordinates(high_pass_spc[idx], positions)
    ampl[idx], mom[idx], wave_fft[idx, :], phase[idx] = fft_fit(wave[idx, :])


fig_high_pass_fit, axes = plt.subplots(3, 3, figsize=(16, 14))

for col in range(3):
    axes[0, col].imshow(high_pass_spc[col, idx_yb:idx_yt, idx_xl:idx_xr],
                        cmap=cm.binary, origin='lower', extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
    axes[0, col].plot([x0, x1], [y0, y1], 'o-', color=u'orange')
    axes[0, col].plot([x_def, x0], [y_def, y0], 'o--', color=u'orange')
    axes[0, col].axis('image')
    axes[0, col].set_title(titles[col])
    axes[0, col].set_xlabel(x_label)

    axes[1, col].plot(k_cut[:idx_cut_4], wave_fft[col, :idx_cut_4], '--o')
    axes[1, col].set_xlabel(r'$k [\mu m^{-1}]$')

    axes[2, col].plot(x_cut, wave[col, :], '-')
    axes[2, col].plot(x_cut, ampl[col] *
                      np.cos(mom[col] * x_cut + phase[col]), '--')
    axes[2, col].set_xlabel(cut_label)

axes[0, 0].set_ylabel(y_label)
axes[1, 0].set_ylabel('magnitude')
fig_high_pass_fit.savefig('fig_exp_high_pass_fit', bbox_inches='tight')

print mom
