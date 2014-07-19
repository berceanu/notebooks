
# coding: utf-8

# <script type="text/javascript">
# on = "Show input";
# off = "Hide input"
# function onoff(){
#   currentvalue = document.getElementById('onoff').value;
#   if(currentvalue == off){
#     document.getElementById("onoff").value=on;
#       $('div.input').hide();
#   }else{
#     document.getElementById("onoff").value=off;
#       $('div.input').show();
#   }
# }
# </script>
# 
# <input type="button" class="ui-button ui-widget ui-state-default
# ui-corner-all ui-button-text-only" value="Hide input" id="onoff"
# onclick="onoff();">

# In[36]:

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from scipy import optimize

from phcpy.phcpy2c import py2c_set_seed
from phcpy.solver import solve
from phcpy.solutions import strsol2dict


# In[37]:

# System parameters
omega_X = omega_C0 = 1.4
kz = 20.0
Omega_R = 2.5*10**(-3)
gamma_X = 0.12*10**(-3)
gamma_C = 0.12*10**(-3)
k_pmpx = 1.6
k_pmpy = 0.0
k_sigx = 0.0
k_sigy = 0.0
k_idlx = 2 * k_pmpx - k_sigx
k_idly = 2 * k_pmpy - k_sigy
omega_pmp = 1.39875
ip_chosen = 6.5
gv = 0.5


# In[38]:

gp = gamma_C + (1 / np.sqrt(1 + (Omega_R / ((0.5 * ((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx**2 + k_pmpy**2) / kz)**2)) + omega_X) - 0.5 * np.sqrt(((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx**2 + k_pmpy**2) / kz)**2)) - omega_X)**2 + 4*Omega_R**2)) - (omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx**2 + k_pmpy**2) / kz)**2))))**2))**2 * (gamma_X - gamma_C)
omega_p_chosen = (omega_pmp - omega_X)/gp


# In[39]:

# Computational parameters

# nkx -> number of columns
# nky -> number of rows
# M[row,col]

nkx = nky = nk = 512
delta_k = 0.05

side_k = nk * delta_k / 2
side_r = np.pi / delta_k
delta_r = np.pi / side_k

x = y = np.arange(-side_r, side_r, delta_r)
kx = ky = np.arange(-side_k, side_k, delta_k)

KX, KY = np.meshgrid(kx, ky)
X, Y = np.meshgrid(x, y)

ipx = np.linspace(0, 15, 30)


# In[40]:

# Update the matplotlib configuration parameters:
# matplotlib.rcParams.update({'font.size': 16, 'font.family': 'serif'})


# In[41]:

#various plotting parameters
x_label_k = r'$k_x[\mu m^{-1}]$'
y_label_k = r'$k_y[\mu m^{-1}]$'
x_label_i = r'$x[\mu m]$'
y_label_i = r'$y[\mu m]$'

letter_spi = ['s', 'p', 'i']
title_k = []
for idx in range(3):
    title_k.append( r'$g \left|\tilde{\Psi}_{LP}^{' + letter_spi[idx] + r'}\left(k+k_{' + letter_spi[idx] +                      r'}\right)\right|^{2} [\gamma_p \mu m^4]$' )
title_i = []
for idx in range(3):
    title_i.append( r'$I^' + letter_spi[idx] + r'$' )

color_spi = ['blue','red','green']

momentum_spi = np.array( [[k_sigx, k_sigy],
                          [k_pmpx, k_pmpy],
                          [k_idlx, k_idly]] )


## Theoretical Model

# The starting GP eq. for the LP band with a $\delta$-like defect and pump is:
# \begin{align}
# i \frac{d}{dt} \widetilde{\Psi}_{\text{LP}}(k)=&\left[\epsilon(k)-i 
# \frac{\gamma(k)}{2}\right]\widetilde{\Psi}_{\text{LP}}(k) + F_p C(k_p) e^{-i 
# \omega_p t} \nonumber \\
# & + \sum_{q_1,q_2} g_{k,q_1,q_2} \widetilde{\Psi}^{*}_{\text{LP}}(q_1+q_2-
# k)\widetilde{\Psi}_{\text{LP}}(q_1)\widetilde{\Psi}_{\text{LP}}(q_2) \nonumber \\
# & + \sum_q G_{k,q} \widetilde{\Psi}_{\text{LP}}(q)
# \end{align}
# 
# with the defect contribution
# \begin{equation}
# G_{k,q} = g_V C(k) C(q)
# \end{equation}
# 
# and the polariton-polariton interaction
# \begin{equation}
# g_{k,q_1,q_2} = g X^*(k) X^*(q_1+q_2-k) X(q_1) X(q_2)
# \end{equation}
# 
# The ansatz for the wavefunction in momentum space reads
# 
# \begin{align}
# \widetilde{\Psi}_{\text{LP}}(k) &= e^{-i\omega_p t} \left[P\delta(k-k_p) + 
# \widetilde{u}_p(k-k_p) e^{-i \omega t} + \widetilde{v}^*_p(k_p-k) e^{i \omega 
# t}\right] \nonumber \\
# & + e^{-i\omega_s t} \left[S\delta(k-k_s) + \widetilde{u}_s(k-k_s) e^{-i \omega 
# t} + \widetilde{v}^*_s(k_s-k) e^{i \omega t}\right] \nonumber \\
# & + e^{-i\omega_i t} \left[I\delta(k-k_i) + \widetilde{u}_i(k-k_i) e^{-i \omega 
# t} + \widetilde{v}^*_i(k_i-k) e^{i \omega t}\right]
# \end{align}
# 
# where we have used the Fourier transforms $u(r)=\sum_k \widetilde{u}(k) e^{ikr}$ 
# and the equivalent for $v$.

# In[42]:

def null(A, eps=1e-10):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


# In[43]:

def index_mom(mom):
    return int( np.floor( (mom + side_k) / delta_k ) )


# In[44]:

def enC(kx,ky):
    return ( omega_C0 * np.sqrt(1 + (np.sqrt(kx**2 + ky**2) / kz)**2) - omega_X ) / gp


# In[45]:

def enLP(kx,ky):
    return 0.5 * enC(kx,ky) - 0.5 * np.sqrt(enC(kx,ky)**2 + 4*Omega_R**2/gp**2)


# In[46]:

def hopf_x(kx,ky):
    return 1 / np.sqrt(1 + ( (Omega_R/gp) / (enLP(kx,ky) - enC(kx,ky)))**2)


# In[47]:

def blue_enLP(kx,ky):
    return enLP(kx,ky) + 2*hopf_x(kx,ky)**2*(ni_chosen + np_chosen + ns_chosen)


# In[48]:

def hopf_c(kx,ky):
    return -1 / np.sqrt(1 + ((enLP(kx,ky) - enC(kx,ky)) / (Omega_R/gp) )**2)


# In[49]:

def gamma(kx,ky):
    return ( gamma_C + hopf_x(kx,ky)**2 * (gamma_X - gamma_C) ) / gp


# In[50]:

def gammaXC(kx,gX,gC):
    return ( gC + hopf_x(kx,0)**2 * (gX - gC) ) / (gp*10**3)


# In[51]:

#some more parameters
#k plotting interval
kl, kr = -8, 8
#kl, kr = -4, 4
kxl, kxr = index_mom(kl), index_mom(kr)

es = enLP(k_sigx,k_sigy) 
ep = enLP(k_pmpx,k_pmpy) 
ei = enLP(k_idlx,k_idly)

gs = gamma(k_sigx,k_sigy)
gi = gamma(k_idlx,k_idly)

xs = hopf_x(k_sigx,k_sigy)
xp = hopf_x(k_pmpx,k_pmpy)
xi = hopf_x(k_idlx,k_idly)

cs = hopf_c(k_sigx, k_sigy)
cp = hopf_c(k_pmpx, k_pmpy)
ci = hopf_c(k_idlx, k_idly)

alpha = (xi**2/xs**2)*(gs/gi)


# In[52]:

print k_idlx, k_idly, gi


# In[53]:

def n_hom_mf(ips):
    return np.array([optimize.brentq(lambda n: ((ep-omega_p_chosen+xp**2*n)**2+1/4)*n - xp**4 * ip, 0,3) for ip in ips])


# In[54]:

def L(kx,ky):
    return np.array( [ [-omega_s_chosen + enLP(k_sigx + kx,k_sigy + ky) - 1j*1/2*gamma(k_sigx + kx,k_sigy + ky) +  2*(ni_chosen + np_chosen + ns_chosen)*hopf_x(k_sigx + kx,k_sigy + ky)**2, 2*(p*np.conjugate(i) + s*np.conjugate(p))*hopf_x(k_pmpx + kx,k_pmpy + ky)*hopf_x(k_sigx + kx,k_sigy + ky), 2*s*np.conjugate(i)*hopf_x(k_idlx + kx,k_idly + ky)*hopf_x(k_sigx + kx,k_sigy + ky), s**2*hopf_x(k_sigx - kx,k_sigy - ky)*hopf_x(k_sigx + kx,k_sigy + ky),2*p*s*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_sigx + kx,k_sigy + ky), (p**2 + 2*i*s)*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_sigx + kx,k_sigy + ky)],                       [2*(i*np.conjugate(p) + p*np.conjugate(s))*hopf_x(k_pmpx + kx,k_pmpy + ky)*hopf_x(k_sigx + kx,k_sigy + ky), -omega_p_chosen + enLP(k_pmpx + kx,k_pmpy + ky) - 1j*1/2*gamma(k_pmpx + kx,k_pmpy + ky) + 2*(ni_chosen + np_chosen + ns_chosen)*hopf_x(k_pmpx + kx,k_pmpy + ky)**2, 2*(p*np.conjugate(i) + s*np.conjugate(p))*hopf_x(k_idlx + kx,k_idly + ky)*hopf_x(k_pmpx + kx,k_pmpy + ky), 2*p*s*hopf_x(k_sigx - kx,k_sigy - ky)*hopf_x(k_pmpx + kx,k_pmpy + ky), (p**2 + 2*i*s)*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_pmpx + kx,k_pmpy + ky), 2*i*p*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_pmpx + kx,k_pmpy + ky)],                       [2*i*np.conjugate(s)*hopf_x(k_idlx + kx,k_idly + ky)*hopf_x(k_sigx + kx,k_sigy + ky), 2*(i*np.conjugate(p) + p*np.conjugate(s))*hopf_x(k_idlx + kx,k_idly + ky)*hopf_x(k_pmpx + kx,k_pmpy + ky), -omega_i_chosen + enLP(k_idlx + kx,k_idly + ky) - 1j*1/2*gamma(k_idlx + kx,k_idly + ky) + 2*(ni_chosen + np_chosen + ns_chosen)*hopf_x(k_idlx + kx,k_idly + ky)**2, (p**2 + 2*i*s)*hopf_x(k_sigx - kx,k_sigy - ky)*hopf_x(k_idlx + kx,k_idly + ky), 2*i*p*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_idlx + kx,k_idly + ky),i**2*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_idlx + kx,k_idly + ky)],                       [-(np.conjugate(s)**2*hopf_x(k_sigx - kx,k_sigy - ky)*hopf_x(k_sigx + kx,k_sigy + ky)), -2*np.conjugate(p)*np.conjugate(s)*hopf_x(k_sigx - kx,k_sigy - ky)*hopf_x(k_pmpx + kx,k_pmpy + ky), -((np.conjugate(p)**2 + 2*np.conjugate(i)*np.conjugate(s))*hopf_x(k_sigx - kx,k_sigy - ky)*hopf_x(k_idlx + kx,k_idly + ky)), omega_s_chosen - enLP(k_sigx - kx,k_sigy - ky) - 1j*1/2*gamma(k_sigx - kx,k_sigy - ky) - 2*(ni_chosen + np_chosen + ns_chosen)*hopf_x(k_sigx - kx,k_sigy - ky)**2, -2*(i*np.conjugate(p) + p*np.conjugate(s))*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_sigx - kx,k_sigy - ky), -2*i*np.conjugate(s)*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_sigx - kx,k_sigy - ky)],                       [-2*np.conjugate(p)*np.conjugate(s)*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_sigx + kx,k_sigy + ky), -((np.conjugate(p)**2 + 2*np.conjugate(i)*np.conjugate(s))*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_pmpx + kx,k_pmpy + ky)), -2*np.conjugate(i)*np.conjugate(p)*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_idlx + kx,k_idly + ky), -2*(p*np.conjugate(i) + s*np.conjugate(p))*hopf_x(k_pmpx - kx,k_pmpy - ky)*hopf_x(k_sigx - kx,k_sigy - ky), omega_p_chosen - enLP(k_pmpx - kx,k_pmpy - ky) - 1j*1/2*gamma(k_pmpx - kx,k_pmpy - ky) - 2*(ni_chosen + np_chosen + ns_chosen)*hopf_x(k_pmpx - kx,k_pmpy - ky)**2, -2*(i*np.conjugate(p) + p*np.conjugate(s))*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_pmpx - kx,k_pmpy - ky)],                       [-((np.conjugate(p)**2 + 2*np.conjugate(i)*np.conjugate(s))*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_sigx + kx,k_sigy + ky)), -2*np.conjugate(i)*np.conjugate(p)*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_pmpx + kx,k_pmpy + ky), -(np.conjugate(i)**2*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_idlx + kx,k_idly + ky)), -2*s*np.conjugate(i)*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_sigx - kx,k_sigy - ky), -2*(p*np.conjugate(i) + s*np.conjugate(p))*hopf_x(k_idlx - kx,k_idly - ky)*hopf_x(k_pmpx - kx,k_pmpy - ky), omega_i_chosen - enLP(k_idlx - kx,k_idly - ky) - 1j*1/2*gamma(k_idlx - kx,k_idly - ky) - 2*(ni_chosen + np_chosen + ns_chosen)*hopf_x(k_idlx - kx,k_idly - ky)**2] ])


# In[55]:

def L_mats(K_X,K_Y,n_kx,n_ky):
    mats = L(K_X,K_Y) 
    new_mats = np.transpose(mats,(2,3,0,1))
    new_mats.shape = (n_kx*n_ky,6,6)
    return new_mats


# In[56]:

def eigL_mats(mats,n_kx,n_ky):
    res = np.linalg.eigvals(mats)
    res.shape = (n_ky,n_kx,6)
    return res


# In[57]:

def fd(kx,ky):
    return np.array([cs/xs * hopf_c(kx + k_sigx, ky + k_sigy) * s,                     cp/xp * hopf_c(kx + k_pmpx, ky + k_pmpy) * p,                     ci/xi * hopf_c(kx + k_idlx, ky + k_idly) * i,                    -cs/xs * hopf_c(k_sigx - kx, k_sigy - ky) * np.conjugate(s),                    -cp/xp * hopf_c(k_pmpx - kx, k_pmpy - ky) * np.conjugate(p),                    -ci/xi * hopf_c(k_idlx - kx, k_idly - ky) * np.conjugate(i)])


# In[58]:

def fd_mats(K_X,K_Y,n_kx,n_ky):
    res = fd(K_X,K_Y)
    new_res = np.transpose(res,(1,2,0))
    new_res.shape = (n_kx*n_ky,6)
    return new_res


# In[59]:

def bog_coef_mats(mats,fds,n_kx,n_ky):
    res = np.linalg.solve(mats, -fds)
    res.shape = (n_ky,n_kx,6)
    return res


## Pump-only stability curve

# $$\left[\left(\epsilon_{p}-\omega_{p}+X_{p}^{2}n_{p}\right)^{2}+\frac{1}{4}\right]n_{p}-X_{p}^{4}I_{p}=0$$

## Mean-field equations

# We start from the mean-field equations

# \begin{equation}
# \begin{aligned}
#     \frac{1}{X_{p}^{2}}\left(\epsilon_{p}-i\frac{\gamma_{p}}{2}-\omega_{p}\right)\tilde{P}+\left(n_{p}+2n_{s}+2n_{i}\right)\tilde{P}+2\tilde{P}^{\star}\tilde{S}\tilde{I}+\tilde{F_{p}}=0 \\
#     \frac{1}{X_{s}^{2}}\left(\epsilon_{s}-i\frac{\gamma_{s}}{2}-\omega_{s}\right)\tilde{S}+\left(2n_{p}+n_{s}+2n_{i}\right)\tilde{S}+\tilde{P}^{2}\tilde{I}^{\star}=0 \\
#     \frac{1}{X_{i}^{2}}\left(\epsilon_{i}-i\frac{\gamma_{i}}{2}-2\omega_{p}+\omega_{s}\right)\tilde{I}+\left(2n_{p}+2n_{s}+n_{i}\right)\tilde{I}+\tilde{P}^{2}\tilde{S}^{\star}=0
# \end{aligned}
# \end{equation}

# where $\tilde{P} = \sqrt{g} X_p P$ etc, $\tilde{F_p} = \sqrt{g} \frac{C_p}{X_p} F_p$ and $n_p = \vert \tilde{P} \vert^2$ etc. We measure energy in units of $\gamma_p$ and $I_{p} = \vert \tilde{F_p} \vert ^2$ in units of $\gamma_p^3$.

# The 4 coupled equations for the unknowns $\omega_{s},n_{s},n_{p},n_i$ as a function of $I_{p}$ are

# \begin{equation}
# \begin{aligned}
#   \left(1-\alpha^{2}\right)n_{s}+2\left(1-\alpha\right)n_{p}-\frac{1}{X_{s}^{2}}\left(1+\frac{\gamma_{s}}{\gamma_{i}}\right)\omega_{s}+\frac{1}{X_{s}^{2}}\left[\epsilon_{s}+\frac{\gamma_{s}}{\gamma_{i}}\left(2\omega_{p}-\epsilon_{i}\right)\right]=0 \\
#   \left[\frac{1}{X_{s}^{2}}\left(\epsilon_{s}-\omega_{s}\right)+2n_{p}+\left(1+2\alpha\right)n_{s}\right]^{2}+\frac{1}{X_{s}^{4}}\frac{\gamma_{s}^{2}}{4}-\alpha n_{p}^{2}=0 \\
#   \left\{ 2n_{s}\left[\frac{1}{X_{s}^{2}}\left(\epsilon_{s}-\omega_{s}\right)+2n_{p}+\left(1+2\alpha\right)n_{s}\right]-n_{p}\left[\frac{1}{X_{p}^{2}}\left(\epsilon_{p}-\omega_{p}\right)+n_{p}+2\left(1+\alpha\right)n_{s}\right]\right\} ^{2}+\frac{1}{4}\left(2\frac{\gamma_{s}}{X_{s}^{2}}n_{s}+\frac{1}{X_{p}^{2}}n_{p}\right)^{2}-n_{p}I_{p}=0 \\
#   n_{i}=\frac{X_{i}^{2}}{X_{s}^{2}}\frac{\gamma_{s}}{\gamma_{i}}n_{s}=\alpha n_{s}
# \end{aligned}
# \end{equation}

# Once the nonlinear system is solved for the unknowns, one can determine $\tilde{P}$ (in units of $\sqrt{\gamma_p}$) from

# $$\tilde{P}\tilde{F_{p}}^{\star}=\frac{2}{X_{s}^{2}}\left(\epsilon_{s}-i\frac{\gamma_{s}}{2}-\omega_{s}\right)n_{s}+2\left(2n_{p}+n_{s}+2n_{i}\right)n_{s}-\left[\frac{1}{X_{p}^{2}}\left(\epsilon_{p}+i\frac{1}{2}-\omega_{p}\right)n_{p}+\left(n_{p}+2n_{s}+2n_{i}\right)n_{p}\right]$$

# where $\tilde{F_p}$ is expressed in units of $\gamma_p \sqrt{\gamma_p}$, and plug it into the following (linear) system in order to determine $\tilde{S}_{r}, \tilde{S}_{i}, \tilde{I}_{r}, \tilde{I}_{i}$

# $$\left(\begin{array}{cccc}
# \frac{1}{X_{s}^{2}}\left(\epsilon_{s}-\omega_{s}\right)+2n_{p}+n_{s}+2n_{i} & \frac{1}{X_{s}^{2}}\frac{\gamma_{s}}{2} & \tilde{P}_{r}^{2}-\tilde{P}_{i}^{2} & 2\tilde{P}_{r}\tilde{P}_{i}\\
# -\frac{1}{X_{s}^{2}}\frac{\gamma_{s}}{2} & \frac{1}{X_{s}^{2}}\left(\epsilon_{s}-\omega_{s}\right)+2n_{p}+n_{s}+2n_{i} & 2\tilde{P}_{r}\tilde{P}_{i} & \tilde{P}_{i}^{2}-\tilde{P}_{r}^{2}\\
# \tilde{P}_{r}^{2}-\tilde{P}_{i}^{2} & 2\tilde{P}_{r}\tilde{P}_{i} & \frac{1}{X_{i}^{2}}\left(\epsilon_{i}-\omega_{i}\right)+2n_{p}+2n_{s}+n_{i} & \frac{1}{X_{i}^{2}}\frac{\gamma_{i}}{2}\\
# 2\tilde{P}_{r}\tilde{P}_{i} & \tilde{P}_{i}^{2}-\tilde{P}_{r}^{2} & -\frac{1}{X_{i}^{2}}\frac{\gamma_{i}}{2} & \frac{1}{X_{i}^{2}}\left(\epsilon_{i}-\omega_{i}\right)+2n_{p}+2n_{s}+n_{i}
# \end{array}\right)\left(\begin{array}{c}
# \tilde{S}_{r}\\
# \tilde{S}_{i}\\
# \tilde{I}_{r}\\
# \tilde{I}_{i}
# \end{array}\right)=0$$

# In[60]:

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


# In[61]:

solutions = []
for pmp_int in ipx:
    solutions.append(eqs(pmp_int))


# In[62]:

solutions_v2 = [sol for sol in solutions if sol != []]


# In[63]:

nsnpip = []
for idx in range(1,4):
    nsnpip.append( np.array([tple[idx] for lst in solutions_v2 for tple in lst]) )


# In[64]:

fig, ax = plt.subplots()

ax.plot(ipx, n_hom_mf(ipx), linestyle='none', marker='o', markerfacecolor='black')

for idx in [0,1]:
    ax.plot(nsnpip[2], nsnpip[idx], linestyle='none', marker='o', markerfacecolor=color_spi[idx])

ax.plot(nsnpip[2], alpha * nsnpip[0], linestyle='none', marker='o', markerfacecolor=color_spi[2])

ax.axvline(x=ip_chosen,color='k',ls='dashed')

ax.set_xlim(ipx[0], ipx[-1])
ax.set_xlabel(r'$I_p [\gamma_p^3]$')
ax.set_ylabel(r'$n_s, n_p [\gamma_p]$')


# In[65]:

[(omega_s_chosen, ns_chosen, np_chosen, ip_chosen)] = eqs(ip_chosen)
omega_i_chosen = 2 * omega_p_chosen - omega_s_chosen
ni_chosen = alpha * ns_chosen


# In[66]:

energy_spi = [omega_s_chosen, omega_p_chosen, omega_i_chosen]


# In[67]:

p = 1/np.sqrt(ip_chosen) * (2/xs**2*(es-1j*gs/2-omega_s_chosen)*ns_chosen+2*(2*np_chosen+ns_chosen+2*ni_chosen)*ns_chosen-(1/xp**2*(ep+1j*1/2-omega_p_chosen)*np_chosen+(np_chosen+2*ns_chosen+2*ni_chosen)*np_chosen))
pr = p.real
pi = p.imag


# In[68]:

matSI = np.array([ [2*ni_chosen + 2*np_chosen + ns_chosen + (es - omega_s_chosen)/xs**2,gs/(2.*xs**2),-pi**2 + pr**2,2*pi*pr],       [-gs/(2.*xs**2),2*ni_chosen + 2*np_chosen + ns_chosen + (es - omega_s_chosen)/xs**2,2*pi*pr,pi**2 - pr**2],       [-pi**2 + pr**2,2*pi*pr,ni_chosen + 2*np_chosen + 2*ns_chosen + (ei - omega_i_chosen)/xi**2,gi/(2.*xi**2)],       [2*pi*pr,pi**2 - pr**2,-gi/(2.*xi**2),ni_chosen + 2*np_chosen + 2*ns_chosen + (ei - omega_i_chosen)/xi**2] ])


# In[69]:

norm = ns_chosen + ni_chosen
N = null(matSI) * np.sqrt(norm)

[sr, si, ir, ii] = N[:,0]
s = (sr + 1j*si)*(1+1j)/np.sqrt(2)
i = (ir + 1j*ii)*(1-1j)/np.sqrt(2)


# In[70]:

print np.abs(np.array([s, p, i]))**2


## Lower polariton dispersion

# In[71]:

gma = np.arange(0.06, 0.20, 0.01)

fig, axes = plt.subplots(1, 2, figsize=(16,3))

for idx in range(3):
    axes[0].plot(gma, gammaXC(momentum_spi[idx,0],0.15,gma), color_spi[idx])
    axes[1].plot(gma, gammaXC(momentum_spi[idx,0],gma,0.12), color_spi[idx])

    axes[0].scatter(0.12, gammaXC(momentum_spi[idx,0],0.15,0.12), c=color_spi[idx])
    axes[1].scatter(0.15, gammaXC(momentum_spi[idx,0],0.15,0.12), c=color_spi[idx])

axes[0].set_xlabel(r'$\gamma_C [meV]$')
axes[1].set_xlabel(r'$\gamma_X [meV]$')

for ax in [0,1]:
    axes[ax].set_xlim(gma[0], gma[-1])
    axes[ax].set_ylabel(r'$\gamma_{s,p,i} [\gamma_p]$')


# In[72]:

LP = enLP(KX, KY)
BLP = blue_enLP(KX, KY)
G = gamma(KX, KY) / 2


# In[73]:

fig, ax = plt.subplots()

ax.plot(KX[nky/2,kxl:kxr], LP[nky/2,kxl:kxr], 'k-.')
ax.fill_between(KX[nky/2,kxl:kxr], LP[nky/2,kxl:kxr] - G[nky/2,kxl:kxr]/2,                LP[nky/2,kxl:kxr] + G[nky/2,kxl:kxr]/2, color='0.75', alpha=0.5)

ax.axhline(y=ep + 0.5 * np.sqrt(3), color='r', ls='dashed')

for idx in range(3):
    ax.scatter(momentum_spi[idx,0], energy_spi[idx], c=color_spi[idx])

ax.set_title('bare LP dispersion')
ax.set_xlim(kl, kr)
ax.set_xlabel(x_label_k)
ax.set_ylabel(r'$\epsilon-\omega_X[\gamma_p]$')


# The blue-shifted LP dispersion is given by $\epsilon\left(k\right)+2\left|X(k)\right|^{2}\sum_{q=1}^{3}n_{q}$.

# In[74]:

fig, ax = plt.subplots()

ax.plot(KX[nky/2,kxl:kxr], BLP[nky/2,kxl:kxr], 'k-.')
ax.fill_between(KX[nky/2,kxl:kxr], BLP[nky/2,kxl:kxr] - G[nky/2,kxl:kxr]/2,                BLP[nky/2,kxl:kxr] + G[nky/2,kxl:kxr]/2, alpha=0.5)

for idx in range(3):
    ax.scatter(momentum_spi[idx,0], energy_spi[idx], c=color_spi[idx])
    
ax.set_title('blue-shifted LP dispersion')
ax.set_xlim(kl, kr)
ax.set_xlabel(x_label_k)
ax.set_ylabel(r'$\epsilon-\omega_X[\gamma_p]$')


## Bogoliubov excitation spectrum

# The spectrum of excitations can be obtained by diagonalizing the Bogoliubov matrix $\mathcal{L}$:

# \begin{equation}
# \mathcal{L}(k) =
# \begin{pmatrix}
# M(k) & Q(k) \\
# -Q^*(-k) & -M^*(-k) 
# \end{pmatrix}
# \end{equation} 

# $$M_{mn}(k)=\left[\epsilon\left(k_{m}+k\right)-\omega_{m}-i\frac{\gamma\left(k_{m}+k\right)}{2}\right]\delta_{m,n}+2X(k_{n}+k)X^{*}(k_{m}+k)\sum_{q,t=1}^{3}\delta_{m+q,n+t}\tilde{A}_{q}^{*}\tilde{A}_{t}$$

# $$Q_{mn}(k)=X^{*}(k_{m}+k)X^{*}(k_{n}-k)\sum_{q,t=1}^{3}\delta_{m+n,q+t}\tilde{A}_{q}\tilde{A}_{t}$$

# with $\tilde{A}_{1}=\tilde{S}$, $\tilde{A}_{2}=\tilde{P}$ and $\tilde{A}_{3}=\tilde{I}$.

### Blue-shifted LP

# In order to get the blue-shifted LP dispersion, we need to set the off-diagonal elements of $\mathcal{L}$ to 0:

# $$M_{ij}(k)=\left[-\omega_{j}+\epsilon\left(k_{j}+k\right)+2\left|X(k_{j}+k)\right|^{2}\sum_{q=1}^{3}n_{q}-\frac{i}{2}\gamma\left(k_{j}+k\right)\right]\delta_{i,j}$$

### Off-diagonal terms in $M$

# With $m\neq n$, we have

# $$M_{mn}(k)=2X(k_{n}+k)X(k_{m}+k)\sum_{r}\tilde{A}_{r}^{\star}\tilde{A}_{r+\left(m-n\right)}=M_{nm}(k)^{\star}$$

# * Interaction between S and I.
# 

# In[75]:

indices_si_rows = [0, 0, 2, 2, 3, 3, 5, 5]
indices_si_cols = [2, 5, 0, 3, 2, 5, 0, 3]


# * Interaction between P and S.

# In[76]:

indices_ps_rows = [0, 0, 1, 1, 3, 3, 4, 4]
indices_ps_cols = [1, 4, 0, 3, 1, 4, 0, 3]


# * Interaction between P and I.

# In[77]:

indices_pi_rows = [1, 1, 2, 2, 4, 4, 5, 5]
indices_pi_cols = [2, 5, 1, 4, 2, 5, 1, 4]


# In[78]:

matsL = L_mats(KX,KY,nkx,nky)
eigs = eigL_mats(matsL,nkx,nky)


# In[79]:

matsL_diag = np.copy(matsL)

matsL_diag[:, indices_si_rows, indices_si_cols] = complex(0,0)
matsL_diag[:, indices_ps_rows, indices_ps_cols] = complex(0,0)
matsL_diag[:, indices_pi_rows, indices_pi_cols] = complex(0,0)

eigs_diag = eigL_mats(matsL_diag,nkx,nky)


# In[80]:

y_points, x_points, eig_indices = np.where(np.abs(eigs.real) <= 0.01)
y_points_diag, x_points_diag, eig_indices_diag = np.where(np.abs(eigs_diag.real) <= 0.01)


# In[81]:

fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(kx[x_points], ky[y_points], marker='o', color='black', s=7, label=r'real $\mathcal{L}$')
ax.scatter(kx[x_points_diag], ky[y_points_diag], marker='^', color='orange', s=7, label=r'diagonal $\mathcal{L}$')

ax.legend()
ax.set_xlim(-8,0)
ax.set_ylim(-4,4)
ax.set_xlabel(x_label_k)
ax.set_ylabel(y_label_k)


# In[82]:

fig, axes = plt.subplots(2,1, figsize=(8,6))

for idx in range(3):
    axes[0].plot(KX[nky/2,kxl:kxr], eigs_diag[nky/2,kxl:kxr,idx].imag, linestyle='none',                 marker='o', markerfacecolor=color_spi[idx], markersize=4)
    axes[0].plot(KX[nky/2,kxl:kxr], eigs[nky/2,kxl:kxr,idx].imag, linestyle='none',                 marker='o', markerfacecolor='black', markersize=4)
    axes[1].plot(KX[nky/2,kxl:kxr], eigs_diag[nky/2,kxl:kxr,idx].real, color=color_spi[idx])
    axes[1].plot(KX[nky/2,kxl:kxr], eigs[nky/2,kxl:kxr,idx].real, linestyle='none',                 marker='o', markerfacecolor='black', markersize=2)

for idx in range(3,6):
    axes[0].plot(KX[nky/2,kxl:kxr], eigs_diag[nky/2,kxl:kxr,idx].imag, linestyle='none',                 marker='o', markerfacecolor=color_spi[idx-3], markersize=4)
    axes[0].plot(KX[nky/2,kxl:kxr], eigs[nky/2,kxl:kxr,idx].imag, linestyle='none',                 marker='o', markerfacecolor='black', markersize=4)
    axes[1].plot(KX[nky/2,kxl:kxr], eigs_diag[nky/2,kxl:kxr,idx].real, linestyle='dashed', color=color_spi[idx-3])
    axes[1].plot(KX[nky/2,kxl:kxr], eigs[nky/2,kxl:kxr,idx].real, linestyle='none',                 marker='o', markerfacecolor='black', markersize=2)
    
for ax in [0,1]:
    axes[ax].axhline(y=0, color='black', ls='dashed')
    axes[ax].axvline(x=0, color='black', ls='dashed')

axes[0].set_ylabel(r'$\Im{(\omega)}[\gamma_p]$')
axes[1].set_xlabel(x_label_k)
axes[0].set_xlim(-0.75, 0.75)
axes[1].set_xlim(kl, kr)
axes[1].set_ylim(-5, 5)
axes[1].set_ylabel(r'$\Re{(\omega)}[\gamma_p]$')


## Momentum-space response

# The response of the system to the perturbation induced by the $\delta$-defect
# is given by $\mathcal{L} \cdot \delta\vec{\psi}_{d}=-\vec{F}_{d}$:

# $$\mathcal{L}(k) \cdot \left(\begin{array}{c}
# u^s_d(k)\\
# u^p_d(k)\\
# u^i_d(k)\\
# v^s_d(k)\\
# v^p_d(k)\\
# v^i_d(k)
# \end{array}\right)=
# -\left(\begin{array}{c}
# \frac{C_{s}}{X_s}C(k+k_{s})\tilde{S}\\
# \frac{C_{p}}{X_p}C(k+k_{p})\tilde{P}\\
# \frac{C_{i}}{X_i}C(k+k_{i})\tilde{I}\\
# -\frac{C_{s}}{X_s}C(k_{s}-k)\tilde{S}^{\star}\\
# -\frac{C_{p}}{X_p}C(k_{p}-k)\tilde{P}^{\star}\\
# -\frac{C_{i}}{X_i}C(k_{i}-k)\tilde{I}^{\star}
# \end{array}\right)$$

# where we have defined $u_d^s = \tilde{u}_d^s \sqrt{g}/g_V$ etc. and we measure $\delta\vec{\psi}_{d}$ in units of $\gamma_p^{-1/2}$ and $g_V$ in units of $\gamma_p \mu m^2$.

# In[83]:

vectfd = fd_mats(KX,KY,nkx,nky)
bcoef = bog_coef_mats(matsL,vectfd,nkx,nky)


# In[84]:

bcoef_diag = bog_coef_mats(matsL_diag,vectfd,nkx,nky)


# $$g \left|\tilde{\Psi}_{\text{LP}}^{p}\left(k+k_{p}\right)\right|^{2}= \left|\tilde{P}/X_{p}\delta(k)+\frac{g_{V}}{2}\left(u_{d}^{p}(k)+v_{d}^{p\star}(-k)\right)\right|^{2}$$

# If we keep just the terms on the diagonal of the matrix $\mathcal{L}$, we can write the response in $k$-space analytically for the signal, pump and idler ($j=1,2,3$):

# $$\left|u_{d}^{j}(k)+v_{d}^{j\star}(-k)\right|^{2}=4\frac{C_{j}^{2}}{X_{j}^{2}}n_{j}\frac{C^{2}(k_{j}+k)}{\left[\epsilon\left(k_{j}+k\right)-\omega_{j}+2 X^2(k_{j}+k) \left(n_{s}+n_{p}+n_{i}\right)\right]^{2}+\frac{1}{4}\gamma^{2}\left(k_{j}+k\right)}$$

# $$\mathcal{L}^{\star}(-k) \cdot \left(\begin{array}{c}
# {u_d^s}^{\star}(-k)\\
# {u_d^p}^{\star}(-k)\\
# {u_d^i}^{\star}(-k)\\
# {v_d^s}^{\star}(-k)\\
# {v_d^p}^{\star}(-k)\\
# {v_d^i}^{\star}(-k)
# \end{array}\right) = - \vec{F}_{d}^{\star}(-k)$$

# In[85]:

matsL_min = -np.fft.fftshift(matsL, axes=(1,2))
vectfd_min = -np.fft.fftshift(vectfd, axes=(1,))
bcoef_conj_mink = bog_coef_mats(matsL_min,vectfd_min,nkx,nky)


# In[86]:

matsL_min_diag = -np.fft.fftshift(matsL_diag, axes=(1,2))
bcoef_conj_mink_diag = bog_coef_mats(matsL_min_diag,vectfd_min,nkx,nky)


# In[87]:

psi_k = gv/2 * (bcoef[:,:,0:3] + bcoef_conj_mink[:,:,3:6])


# In[88]:

psi_k_diag = gv/2 * (bcoef_diag[:,:,0:3] + bcoef_conj_mink_diag[:,:,3:6])


### Averaging in the center

# In[89]:

N = 15
# make an empty data set
data = np.ones((N, N)) * np.nan
# fill in some fake data
for idx in range(3)[::-1]:
    data[N//2 - idx : N//2 + idx +1, N//2 - idx : N//2 + idx +1] = idx
# make color map
my_cmap = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
# set the 'bad' values (nan) to be white and transparent
my_cmap.set_bad(color='w', alpha=0)
# draw the grid
for idx in range(N + 1):
    plt.axhline(idx, lw=2, color='k', zorder=5)
    plt.axvline(idx, lw=2, color='k', zorder=5)
# draw the boxes
plt.axis('off')
plt.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)


# In[90]:

[imax, jmax] = np.unravel_index(np.argmax(np.abs(psi_k[:,:,0])), psi_k[:,:,0].shape)
print imax, jmax


# In[91]:

l1 = psi_k[imax - 2 : imax + 3, jmax - 2,:]
l2 = psi_k[imax - 2 : imax + 3, jmax + 2,:]
l3 = psi_k[    imax - 2,    jmax - 1 : jmax + 2,:]
l4 = psi_k[    imax + 2,    jmax - 1 : jmax + 2,:]

l = np.concatenate((l1, l2, l3, l4))
averages = np.mean(l, axis=0)
psi_k[imax - 1 : imax + 2, jmax - 1 : jmax + 2,:] = averages


# In[92]:

#add constant part to wavefunction at k = 0
psi_k[nky/2, nkx/2,:] += np.sqrt(nkx*nky) * np.array([s/xs,p/xp,i/xi])


# In[93]:

#add constant part to wavefunction at k = 0
psi_k_diag[nky/2, nkx/2,:] += np.sqrt(nkx*nky) * np.array([s/xs,p/xp,i/xi])


# In[94]:

res_k = np.log10(np.abs(psi_k)**2) #logscale


# We first plot the full response in momentum-space, corresponding to the Bogoliubov matrix $\mathcal{L}$.

# In[95]:

fig, axes = plt.subplots(1,3, figsize=(14,14))

for idx in range(3):
    axes[idx].imshow(res_k[kxl:kxr,kxl:kxr,idx],                     cmap=cm.gray, origin=None, extent=[kx[kxl], kx[kxr], ky[kxl], ky[kxr]])
    axes[idx].set_title(title_k[idx])

axes[0].set_ylabel(y_label_k)    
for ax in range(3):
    axes[ax].set_xlabel(x_label_k)


# The next series of 3 contour plots correspond to the response calculated by keeping only the terms on the S-S, P-P and I-I interactions and setting all other terms in $\mathcal{L}$ to 0.

# In[96]:

res_k_diag = np.log10(np.abs(psi_k_diag)**2) #logscale


# In[97]:

fig, axes = plt.subplots(1,3, figsize=(14,14))

for idx in range(3):
    axes[idx].imshow(res_k_diag[kxl:kxr,kxl:kxr,idx],                     cmap=cm.gray, origin=None, extent=[kx[kxl], kx[kxr], ky[kxl], ky[kxr]])
    axes[idx].set_title(title_k[idx])

axes[0].set_ylabel(y_label_k)    
for ax in range(3):
    axes[ax].set_xlabel(x_label_k)


# In[98]:

kl3d, kr3d = -5, 5
kxl3d, kxr3d = index_mom(kl3d), index_mom(kr3d)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_surface(KX[kxl3d:kxr3d,kxl3d:kxr3d], KY[kxl3d:kxr3d,kxl3d:kxr3d], BLP[kxl3d:kxr3d,kxl3d:kxr3d],                rstride=8, cstride=8, alpha=0.4)

for idx in range(3):
    ax.plot(kx[x_points] + momentum_spi[idx,0], ky[y_points] + momentum_spi[idx,1], energy_spi[idx],            linestyle='none', marker='o', markerfacecolor=color_spi[idx], markersize=5)

#ax.view_init(0, -90)

ax.set_xlim(kl3d, kr3d)
ax.set_ylim(kl3d, kr3d)

ax.set_xlabel(r'$k_x[\mu m^{-1}]$')
ax.set_ylabel(r'$k_y[\mu m^{-1}]$')
ax.set_zlabel(r'$\epsilon-\omega_X[\gamma_p]$')


## Real-space response

# The last step is to transform to real-space, according to

# $$I^{p}(r)=\frac{\vert\Psi_{\text{LP}}^{p}(r)\vert^{2}}{\vert P\vert^{2}}=\frac{\left|\sum_{k}\left[\tilde{P}/X_{p}\delta(k)+g_V/2\left(u_{d}^{p}(k)+v_{d}^{p\star}(-k)\right)\right]e^{ikr}\right|^{2}}{n_p/X_p^2}$$

# In[99]:

psi_r = np.fft.fftshift(np.fft.ifft2(np.sqrt(nkx*nky) * psi_k, axes=(0,1)), axes=(0,1))
res_r = np.abs(psi_r)**2 / np.array([ns_chosen/xs**2, np_chosen/xp**2, ni_chosen/xi**2])


# We plot the full response in real-space, corresponding to the Bogoliubov matrix $\mathcal{L}$.

# In[100]:

fig, axes = plt.subplots(1,3, figsize=(24,24))

for idx in range(3):
    axes[idx].imshow(res_r[:,:,idx], cmap=cm.gray, origin=None, extent=[x[0],x[-1],y[0],y[-1]])
    axes[idx].set_title(title_i[idx])

axes[0].set_ylabel(y_label_i)
for ax in range(3):
    axes[ax].set_xlabel(x_label_i)


# In[101]:

# ranges of the above plots
for idx in range(3):
    print np.amin(res_r[:,:,idx]), np.amax(res_r[:,:,idx])


# In[102]:

psi_r_diag = np.fft.fftshift(np.fft.ifft2(np.sqrt(nkx*nky) * psi_k_diag, axes=(0,1)), axes=(0,1))
res_r_diag = np.abs(psi_r_diag)**2 / np.array([ns_chosen/xs**2, np_chosen/xp**2, ni_chosen/xi**2])


# The next series of 3 contour plots correspond to the response calculated by keeping only the S-S, P-P and I-I terms of $\mathcal{L}$, setting all others to 0.

# In[103]:

fig, axes = plt.subplots(1,3, figsize=(24,24))

for idx in range(3):
    axes[idx].imshow(res_r_diag[:,:,idx], cmap=cm.gray, origin=None, extent=[x[0],x[-1],y[0],y[-1]])
    axes[idx].set_title(title_i[idx])

axes[0].set_ylabel(y_label_i)
for ax in range(3):
    axes[ax].set_xlabel(x_label_i)


# In[104]:

(# ranges of the above plots
for idx in range(3):
    print np.amin(res_r_diag[:,:,idx]), np.amax(res_r_diag[:,:,idx])

