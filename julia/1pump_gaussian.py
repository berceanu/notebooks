
# coding: utf-8

# In[1]:

using PyPlot
using Polynomial


# In[2]:

# System parameters
#energies in eV
global const omega_X = omega_C0 = 1.483952
global const kz = 20. #μm^-1
global const Omega_R = 3e-3
global const gamma_X = 6.582119272010504e-7
global const gamma_C = 0.0001316423854402101
global const k_pmpx = 0.89
global const k_pmpy = 0.
global const gp = gamma_C + (1 / sqrt(1 + (Omega_R / ((0.5 * ((omega_C0 * sqrt(1 + (sqrt(k_pmpx^2 + k_pmpy^2) / kz)^2)) + omega_X) - 0.5 * sqrt(((omega_C0 * sqrt(1 + (sqrt(k_pmpx^2 + k_pmpy^2) / kz)^2)) - omega_X)^2 + 4Omega_R^2)) - (omega_C0 * sqrt(1 + (sqrt(k_pmpx^2 + k_pmpy^2) / kz)^2))))^2))^2 * (gamma_X - gamma_C)
global const omega_p = (1.48283 - omega_X)/gp


# We measure energies in units of $\gamma_p$, with the origin set to $\omega_X$.

# In[3]:

enC(kx,ky) = (omega_C0 * sqrt(1. + (sqrt(kx^2 + ky^2) / kz)^2) - omega_X) / gp
enLP(kx,ky) = 0.5 * enC(kx,ky) - 0.5 * sqrt(enC(kx,ky)^2 + 4.Omega_R^2/gp^2)
hopf_x(kx,ky) = 1. / sqrt(1. + ((Omega_R/gp) / (enLP(kx,ky) - enC(kx,ky)))^2)
gamma(kx,ky) = (gamma_C + hopf_x(kx,ky)^2 * (gamma_X - gamma_C)) / gp


# In[4]:

hopf_c(kx,ky) = -1 / sqrt(1 + ((enLP(kx,ky) - enC(kx,ky)) / (Omega_R/gp) )^2)


# The blue-shifted LP dispersion is given by $\epsilon\left(k\right)+2n_{p}\left|X(k)\right|^{2}$.

# In[5]:

bsenLP(kx,ky,np_chosen) = enLP(kx,ky) + 2np_chosen*abs2(hopf_x(kx, ky))


# In[6]:

# exciton-exciton contact interaction in γp*μm^2  
gX = 5e-6/gp
# experimental blue-shift, in units of γp
bshift = 0.0012924472535611464/gp
#global const n_p = bshift/2abs2(hopf_x(k_pmpx , k_pmpy))
global const n_p = 43.
# alternatively
#global const n_p = gX*abs2(hopf_x(k_pmpx, k_pmpy))*650


# $$n_{p}=g\left|X_{p}\right|^{2}\left|\psi_{p}^{\text{ss}}\right|^{2}$$

# In[7]:

# inferred exp pump polariton density, in μm^-2
n_p/(gX*abs2(hopf_x(k_pmpx, k_pmpy)))


# In[8]:

# Computational parameters
global const kx = linspace(-12.8, 12.75, 512);


# In[9]:

blp = Float64[bsenLP(k, 0., n_p) for k in kx];
gma = Float64[gamma(k, 0.) for k in kx]/2;


# In[10]:

fig, ax = plt.subplots(figsize=(5, 3))

ax[:plot](kx, blp, "k-.")
ax[:fill_between](kx, blp - gma/2, blp + gma/2, alpha=0.5)

ax[:scatter](k_pmpx, omega_p)
    
ax[:set_title]("blue-shifted LP dispersion")
ax[:set_xlim](-8, 8)
ax[:set_xlabel](L"$k_x[\mu m^{-1}]$")
ax[:set_ylabel](L"$\epsilon-\omega_X[\gamma_p]$")


# $$M(k)=\epsilon(k_{p}+k)-\omega_{p}-\frac{i}{2}\gamma(k_{p}+k)+2n_{p}\left|X(k_{p}+k)\right|^{2}$$
# $$Q(k)=n_{p}X^{*}(k_{p}+k)X^{*}(k_{p}-k)$$

# In[11]:

M(kx, ky, np) = enLP(k_pmpx + kx, k_pmpy + ky) - omega_p - im*gamma(k_pmpx + kx, k_pmpy + ky)/2 + 2np*abs2(hopf_x(k_pmpx + kx, k_pmpy + ky))
Q(kx, ky, np) = np*conj(hopf_x(k_pmpx + kx, k_pmpy + ky))*conj(hopf_x(k_pmpx - kx, k_pmpy - ky))


# $$L(k)=\left(\begin{matrix}M\left(k\right) & Q\left(k\right)e^{2i\phi_{p}}\\
# -e^{-2i\phi_{p}}Q^{*}\left(-k\right) & -M^{*}\left(-k\right)
# \end{matrix}\right)$$

# In[12]:

L(kx, ky, np, φp) = [M(kx, ky, np) Q(kx, ky, np)*exp(2im*φp); -exp(-2im*φp)*conj(Q(-kx, -ky, np)) -conj(M(-kx, -ky, np))]


# $$w(k)=M\left(k\right)-M^{*}\left(-k\right)$$
# $$z(k)=\left[M\left(k\right)+M^{*}\left(-k\right)\right]^{2}-4Q\left(k\right)Q^{*}\left(-k\right)$$

# In[13]:

w(kx, ky, np) = M(kx, ky, np) - conj(M(-kx, -ky, np))
z(kx, ky, np) = (M(kx, ky, np) + conj(M(-kx, -ky, np)))^2 - 4Q(kx, ky, np)*conj(Q(-kx, -ky, np))


# $$\lambda\left(k\right)_{1,2}=\frac{1}{2}w\pm\frac{1}{2}\sqrt{z}$$

# In[14]:

λ1(kx, ky, np) = 1/2*w(kx, ky, np) + 1/2*sqrt(z(kx, ky, np))
λ2(kx, ky, np) = 1/2*w(kx, ky, np) - 1/2*sqrt(z(kx, ky, np))


# In[15]:

np = n_p
fig, axes = plt.subplots(2,1, figsize=(8,6))
for i = 1:2
    axes[1][:scatter](kx, real([λ1(k, 0., np) for k in kx]), s=15, alpha=0.4, color="orange")
    axes[1][:scatter](kx, real([λ2(k, 0., np) for k in kx]), s=15, alpha=0.2)
    axes[2][:scatter](kx, imag([λ1(k, 0., np) for k in kx]), s=15, alpha=0.4, color="orange")
    axes[2][:scatter](kx, imag([λ2(k, 0., np) for k in kx]), s=15, alpha=0.2)
end
for ax in axes
    ax[:set_xlim](kx[1], kx[end])
    ax[:grid]()
end
axes[1][:set_ylabel](L"$\Re(\omega)[\gamma_p]$")
axes[2][:set_ylabel](L"$\Im(\omega)[\gamma_p]$")
axes[2][:set_xlabel](L"$k_x[\mu m^{-1}]$")


# $$\left|X_{p}\right|^{4}n_{p}^{3}+2\left|X_{p}\right|^{2}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}-\left|X_{p}\right|^{4}I_{p}=0$$

# In[16]:

function mfroots(kpx::Float64, kpy::Float64, ωp::Float64, Ip::Float64)
    xp = hopf_x(kpx, kpy)
    ep = enLP(kpx, kpy)
    a = abs2(xp)^2
    b = 2abs2(xp)*(ep - ωp)
    c = 1/4 + (ep - ωp)^2
    d = -a*Ip
    r = roots(Poly([a, b, c, d]))
    filter!(x -> isapprox(imag(x), 0), r)
    rr = real(r)
    shade = Array(ASCIIString, size(rr))
    fill!(shade, "blue")
    for idx = 1:length(rr)
        np = rr[idx]
        for kx = -5:0.05:5
            if imag(λ1(kx, 0., np)) > 0 || imag(λ2(kx, 0., np)) > 0
                shade[idx] = "red"
                break
            end
        end
    end
    ips = Array(Float64, size(rr))
    fill!(ips, Ip)
    return (ips, rr, shade)
end


# $$n_{p}^{3}+\frac{2}{\left|X_{p}\right|^{2}}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\frac{1}{\left|X_{p}\right|^{4}}\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}=I_{p}$$

# In[17]:

function findpump(kpx::Float64, kpy::Float64, ωp::Float64, np::Float64)
    xp = hopf_x(kpx, kpy)
    ep = enLP(kpx, kpy)
    b = 2/abs2(xp)*(ep - ωp)
    c = 1/abs2(xp)^2*(1/4 + (ep - ωp)^2)
    np^3 + b*np^2 + c*np
end


# In[18]:

ipchosen = findpump(k_pmpx, k_pmpy, omega_p, n_p)


# In[20]:

fig, ax = plt.subplots(figsize=(8, 3))
ipmax=2ipchosen
for ip in linspace(0, ipmax, 100)
    ips, nps, colors = mfroots(k_pmpx, k_pmpy, omega_p, ip)
    for idx = 1:length(ips)
        ax[:scatter](ips[idx], nps[idx], s=15, alpha=0.4, color=colors[idx])
    end
end
ax[:axvline](x=ipchosen, color="black", ls="dashed")
ax[:grid]()
ax[:set_ylim](0, 70)
ax[:set_xlim](0, ipmax)
ax[:set_xlabel](L"$I_p [\gamma_p^3]$")
ax[:set_ylabel](L"$n_p [\gamma_p]$")


# In[21]:

# nss    -->   np
#325:975 --> 40:59.64


# In[19]:

#check
mfroots(k_pmpx, k_pmpy, omega_p, ipchosen)


# $$V_d(k) = g_V e^{-k^2 \sigma^2}$$

# In[22]:

Vd(kx, ky, σ, gV) = gV*exp(-σ^2*(kx^2+ky^2))


# In[65]:

vd = [Vd(kx, ky, 10, 0.5) for kx=-8:0.05:8, ky=-8:0.05:8];


# In[66]:

plt.imshow(vd, ColorMap("gray"), origin="lower",
                      extent=[-8, 8, -8, 8])
plt.axis("image")
plt.title("defect potential")
plt.ylim(-8, 8)
plt.xlim(-8, 8)
plt.xlabel(L"$k_x [\mu m^{-1}]$")
plt.ylabel(L"$k_y [\mu m^{-1}]$")


# $$R(k) = \frac{C(k_p)}{X(k_p)}C(k+k_p)$$

# In[25]:

R(kx, ky) = hopf_c(k_pmpx, k_pmpy)/hopf_x(k_pmpx, k_pmpy)*hopf_c(k_pmpx + kx, k_pmpy + ky)


# $$\left|\widetilde{\psi}\left(k+k_{p}\right)\right|^{2}=\frac{n_{p}}{g}\left|\frac{\delta(k)}{X_{p}}+V_{d}\left(k\right)\frac{Q\left(k\right)R^{*}\left(-k\right)-M^{*}\left(-k\right)R\left(k\right)}{M\left(k\right)M^{*}\left(-k\right)-Q\left(k\right)Q^{*}\left(-k\right)}\right|^{2}$$

# In[26]:

ψtmom(kx, ky, np, σ, gV) = Vd(kx, ky, σ, gV)*(Q(kx, ky, np)*conj(R(-kx, -ky)) - conj(M(-kx, -ky, np))*R(kx, ky))/(M(kx, ky, np)*conj(M(-kx, -ky, np)) - Q(kx, ky, np)*conj(Q(-kx, -ky, np)))


# In[28]:

ψmom = Complex{Float64}[ψtmom(momx, momy, n_p, 0.2, 0.5) for momx in kx, momy in kx];


# In[50]:

kx[257]


# In[30]:

#add constant part to wavefunction at k = 0
ψmom[257, 257] += 512/hopf_x(k_pmpx, k_pmpy);


# In[52]:

kx'


# In[59]:

ψr = abs2(hopf_x(k_pmpx, k_pmpy)*fftshift(ifft(512ψmom)));


# In[62]:

fig_real_P, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[1][:imshow](ψr', ColorMap("gray"),
                 origin=None)
axes[2][:plot](ψr[:, 257])
axes[2][:grid]()
#axes[1][:set_ylabel](y_label_i)
#axes[1][:set_xlabel](x_label_i)
#axes[2][:set_xlabel](x_label_i)


# $$I(r)=\frac{\vert\psi(r)\vert^{2}}{\vert\psi_p^{\text{ss}}\vert^{2}}=\left|X_{p}\right|^{2}\left|\sum_{k}\left[\frac{\delta(k)}{X_{p}}+V_{d}\left(k\right)\frac{Q\left(k\right)R^{*}\left(-k\right)-M^{*}\left(-k\right)R\left(k\right)}{M\left(k\right)M^{*}\left(-k\right)-Q\left(k\right)Q^{*}\left(-k\right)}\right]e^{ikr}\right|^{2}$$

# In[49]:

fig_mom_P, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[1][:imshow](log10(n_p*abs2(ψmom))',
ColorMap("gray"), origin=None, extent=[kx[1], kx[end], kx[1], kx[end]])
axes[2][:plot](kx, log10(n_p*abs2(ψmom[:,257]))) 
axes[2][:grid]()
axes[1][:set_ylabel](L"$k_y [\mu m^{-1}]$")
axes[1][:set_xlabel](L"$k_x [\mu m^{-1}]$")
axes[2][:set_xlabel](L"$k_x [\mu m^{-1}]$")


# In[32]:

#linspace(-12.8, 12.75, 512)


# In[33]:

kx[257]


# In[34]:



