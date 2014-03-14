import opo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import ipdb

def a(m):
    return -np.pi * m


def x0(m, k_0):
    return np.pi * m / k_0


mpl.rcParams.update({'font.size': 22, 'font.family': 'serif'})
x_label = r'$x[\mu m]$'
y_label = r'$y[\mu m]$'
titles = ['signal', 'pump', 'idler']


ib = 115
it = 137
il = 501
ir = 523
tukey_alpha = 0.3

ny = 250
nx = 1024
mask = np.zeros((ny, nx))
delta_y = 0.64
delta_x = 0.16
side_y = ny * delta_y / 2
side_x = nx * delta_x / 2
y = np.arange(-side_y, side_y, delta_y)
x = np.arange(-side_x, side_x, delta_x)

x_def = 14.
y_def = -5.
side = 35
xl =-18 #x_def - side
xr = 49 #x_def + side
yb = -38 #y_def - side
yt = 28 #y_def + side
idx_xl = opo.find_index(xl, side_x, delta_x)
idx_xr = opo.find_index(xr, side_x, delta_x)
idx_yb = opo.find_index(yb, side_y, delta_y)
idx_yt = opo.find_index(yt, side_y, delta_y)


k0 = 0.9 * 2
t = np.arange(-10, 10, 0.1)


home = os.getenv("HOME")
base = home + '/ownCloud/Dropbox/experiments_OPO-defect/'
folder = 'OPO_exp_2012_aug_29_data+fig_new/'
path_s = base + folder + 'an_OPO-serie01-signal-original_590_04.dat'
path_p = base + folder + 'an_OPO-serie01-pump-original_636_04.dat'
path_i = base + folder + 'an_OPO-serie01-idler-original_682_04.dat'
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


#ipdb.set_trace()

fig_high_pass_parabola_s, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.fliplr(np.rot90(high_pass_spc[0, idx_yb:idx_yt, idx_xl:idx_xr])),
                 cmap=cm.gray, origin='lower',
                 extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
for M in range(1, 10):
    x_t = x_def - a(M) * t ** 2  - x0(M, k0)
    y_t = y_def + 2 * a(M) * t
    ax.plot(x_t, y_t)
ax.scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
ax.axis('image')
ax.set_ylim(y[idx_yb], y[idx_yt])
ax.set_xlim(x[idx_xl], x[idx_xr])
#ax.set_title(titles[0])
#ax.set_xlabel(x_label)
ax.set_xticklabels([])
ax.set_ylabel(y_label)
fig_high_pass_parabola_s.savefig('fig_exp_high_pass_parabola_s', bbox_inches='tight')

fig_high_pass_parabola_p, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.fliplr(np.rot90(high_pass_spc[1, idx_yb:idx_yt, idx_xl:idx_xr])),
                 cmap=cm.gray, origin='lower',
                 extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
for M in range(1, 10):
    x_t = x_def - a(M) * t ** 2  - x0(M, k0)
    y_t = y_def + 2 * a(M) * t
    ax.plot(x_t, y_t)
ax.scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
ax.axis('image')
ax.set_ylim(y[idx_yb], y[idx_yt])
ax.set_xlim(x[idx_xl], x[idx_xr])
#ax.set_title(titles[1])
#ax.set_xlabel(x_label)
ax.set_xticklabels([])
ax.set_ylabel(y_label)
fig_high_pass_parabola_p.savefig('fig_exp_high_pass_parabola_p', bbox_inches='tight')

fig_high_pass_parabola_i, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.fliplr(np.rot90(high_pass_spc[2, idx_yb:idx_yt, idx_xl:idx_xr])),
                 cmap=cm.gray, origin='lower',
                 extent=[x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]])
for M in range(1, 10):
    x_t = x_def - a(M) * t ** 2  - x0(M, k0)
    y_t = y_def + 2 * a(M) * t
    ax.plot(x_t, y_t)
ax.scatter(x_def, y_def, s=40, c=u'orange', marker=u'o')
ax.axis('image')
ax.set_ylim(y[idx_yb], y[idx_yt])
ax.set_xlim(x[idx_xl], x[idx_xr])
#ax.set_title(titles[2])
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
fig_high_pass_parabola_i.savefig('fig_exp_high_pass_parabola_i', bbox_inches='tight')
