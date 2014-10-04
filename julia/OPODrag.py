from phcpy.phcpy2c import py2c_set_seed
from phcpy.solver import solve
from phcpy.solutions import strsol2dict


alpha = (xi ** 2 / xs ** 2) * (gs / gi)


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
s = sr + 1j * si
i = ir + 1j * ii

#####
########

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


psi_r = np.fft.fftshift(
    np.fft.ifft2(np.sqrt(nkx * nky) * psi_k, axes=(0, 1)), axes=(0, 1))
res_r = np.abs(psi_r) ** 2 / \
    np.array([ns_chosen / xs ** 2, np_chosen / xp ** 2, ni_chosen / xi ** 2])


np.save("/home/berceanu/notebooks/OPODrag/res_r", res_r)
print "done!"
