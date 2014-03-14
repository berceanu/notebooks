import opo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


def a(m):
    return -np.pi * m


def x0(m, k_0):
    return np.pi * m / k_0


mpl.rcParams.update({'font.size': 16, 'font.family': 'serif'})
x_label = r'$x[\mu m]$'
y_label = r'$y[\mu m]$'
titles = ['signal', 'pump', 'idler']


ib = 115
it = 137
il = 501
ir = 523
tukey_alpha = 0.3

ny = 200
nx = 1024
mask = np.zeros((ny, nx))
delta_y = 0.64
delta_x = 0.16
side_y = ny * delta_y / 2
side_x = nx * delta_x / 2
y = np.arange(-side_y, side_y, delta_y)
x = np.arange(-side_x, side_x, delta_x)

x_def = -9.5
y_def = 1.
side = 35
xl = x_def - side
xr = x_def + side
yb = y_def - side
yt = y_def + side
idx_xl = opo.find_index(xl, side_x, delta_x)
idx_xr = opo.find_index(xr, side_x, delta_x)
idx_yb = opo.find_index(yb, side_y, delta_y)
idx_yt = opo.find_index(yt, side_y, delta_y)


k0 = 0.8 * 2
t = np.arange(-10, 10, 0.1)


home = os.getenv("HOME")
base = home + '/ownCloud/Dropbox/experiments_OPO-defect/'
folder = 'OPO-exp-20130419_int_maps_data/'
path_s = base + folder + 'E-signal.dat'
path_p = base + folder + 'E-pump.dat'
path_i = base + folder + 'E-idler.dat'
data_s = np.loadtxt(path_s)
data_p = np.loadtxt(path_p)
data_i = np.loadtxt(path_i)
data = np.array([data_s, data_p, data_i])
data_fft = np.fft.fftshift(np.fft.fft2(data), axes=(1, 2))


tukey_length = data_fft[1, ib:it, il:ir].shape[-1]
tw_x = tw_y = opo.tukeywin(tukey_length, tukey_alpha)
tw_2d = np.outer(tw_y, tw_x)
mask[ib:it, il:ir] = tw_2d
low_pass = data_fft * mask
high_pass = data_fft - low_pass
high_pass_spc = np.real(np.fft.ifft2(np.fft.ifftshift(high_pass, axes=(1, 2))))


fig_high_pass_parabola, axes = plt.subplots(1, 3, figsize=(15, 5))
for col in range(3):
    axes[col].imshow(high_pass_spc[col, idx_yb:idx_yt, idx_xl:idx_xr],
                     cmap=cm.gray, origin='lower',
                     extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
    for M in range(1, 8):
        x_t = -a(M) * t ** 2 - x0(M, k0) + x_def
        y_t = 2 * a(M) * t + y_def
        axes[col].plot(x_t, y_t)
    axes[col].scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
    axes[col].axis('image')
    axes[col].set_ylim(y[idx_yb], y[idx_yt])
    axes[col].set_xlim(x[idx_xl], x[idx_xr])
    axes[col].set_title(titles[col])
    axes[col].set_xlabel(x_label)
axes[0].set_ylabel(y_label)
fig_high_pass_parabola.savefig('fig_exp_high_pass_parabola_april', bbox_inches='tight')
