data = np.array([data_s, data_s, data_s])
data.shape = ((-1, Ny, Nx))

def_x_pos = 11.
def_y_pos = -0.546875
idx_def_x_pos = find_index(def_x_pos, Lx, a_x)
idx_def_y_pos = find_index(def_y_pos, Ly, a_y)
x_def = x[idx_def_x_pos]
y_def = y[idx_def_y_pos]

side = 15. /   # half-side of square centered on defect
xl = x_def - side
xr = x_def + side
yb = y_def - side
yt = y_def + side
idx_xl = find_index(xl, Lx, a_x)
idx_xr = find_index(xr, Lx, a_x)
idx_yb = find_index(yb, Ly, a_y)
idx_yt = find_index(yt, Ly, a_y)

my_ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=a_y))
my_kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=a_x))

data_fft = np.fft.fftshift(np.fft.fft2(data), axes=(1, 2))

###
fig_data_fft, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.log10(np.abs(data_fft[0, :, :])),
        cmap=cm.gray, origin='lower',
        extent=np.array([my_kx[0], my_kx[-1],
            my_ky[0], my_ky[-1]]) / )
ax.set_ylabel(ky_label)
ax.set_xlabel(kx_label)
fig_data_fft.savefig('fig_data_fft', bbox_inches='tight')
###

kyb = kxl = - 2.
kyt = kxr = 2.

idx_kyb = find_index(kyb, np.abs(my_ky[0]), my_ky[1] - my_ky[0])
idx_kyt = find_index(kyt, np.abs(my_ky[0]), my_ky[1] - my_ky[0])
idx_kxl = find_index(kxl, np.abs(my_kx[0]), my_kx[1] - my_kx[0])
idx_kxr = find_index(kxr, np.abs(my_kx[0]), my_kx[1] - my_kx[0])


###
fig_data_fft_cut, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.log10(
    np.abs(data_fft[0, idx_kyb:idx_kyt, idx_kxl:idx_kxr])),
    cmap=cm.gray, origin='lower',
    extent=np.array( [my_kx[idx_kxl], my_kx[idx_kxr],
        my_ky[idx_kyb], my_ky[idx_kyt]]) / )
ax.set_ylabel(ky_label)
ax.set_xlabel(kx_label)
fig_data_fft_cut.savefig('fig_data_fft_cut',
        bbox_inches='tight')
###

il = ib = 72
ir = it = 184

tukey_length = data_fft[1, ib:it, il:ir].shape[-1]
tukey_alpha = 0.6
tw_x = tw_y = tukeywin(tukey_length, tukey_alpha)
tw_2d = np.outer(tw_y, tw_x)

###
fig_tukey, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(my_kx[il:ir] / , tw_x, '-^')
ax.set_xlabel(kx_label)
ax.set_xlim(0., my_kx[ir] / )
ax.set_ylim(0., 1.1)
fig_tukey.savefig('fig_tukey', bbox_inches='tight')
###

mask = np.zeros((Ny, Nx))
mask[ib:it, il:ir] = tw_2d

low_pass = data_fft * mask
high_pass = data_fft - low_pass

high_pass_spc = np.real(np.fft.ifft2(np.fft.ifftshift(high_pass, axes=(1, 2))))

###
fig_high_pass_s, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(high_pass_spc[0, idx_yb:idx_yt, idx_xl:idx_xr],
                 cmap=cm.gray, origin='lower', extent=np.array([x[idx_xl], x[idx_xr], y[idx_yb], y[idx_yt]]) * )
ax.scatter(x_def * , y_def * , s=50, c=u'orange', marker=u'o')
ax.axis('image')
ax.set_ylabel(y_label)
ax.set_xlabel(x_label)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim( * x[idx_xl],  * x[idx_xr])
ax.set_ylim( * y[idx_yb],  * y[idx_yt])
fig_high_pass_s.savefig('fig_high_pass_s', bbox_inches='tight')
###
