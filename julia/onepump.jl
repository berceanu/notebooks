module onepump

using PyPlot
using Polynomial


# System parameters
#energies in eV
global const ωx = ωc = 1.483952
global const kz = 20. #μm^-1
global const Ωr = 3e-3
global const γx = 6.582119272010504e-7
global const γc = 0.0001316423854402101
global const kpx = 0.89
global const kpy = 0.
global const γp = γc + (1/sqrt(1+(Ωr/((1/2*((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))+ωx)-1/2*sqrt(((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))-ωx)^2+4Ωr^2))
                - (ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))))^2))^2*(γx-γc)
global const ωp = (1.48283 - ωx)/γp


# We measure energies in units of $\gamma_p$, with the origin set to $\omega_X$.
enc(ky::Float64, kx::Float64) = (ωc * sqrt(1 + (sqrt(kx^2 + ky^2)/kz)^2) - ωx)/γp
enlp(ky::Float64, kx::Float64) = 1/2*enc(ky, kx) - 1/2*sqrt(enc(ky, kx)^2 + 4Ωr^2/γp^2)
hopfx(ky::Float64, kx::Float64) = 1/sqrt(1+((Ωr/γp) / (enlp(ky, kx) - enc(ky, kx)))^2)
hopfc(ky::Float64, kx::Float64) = -1/sqrt(1+((enlp(ky, kx) - enc(ky, kx)) /(Ωr/γp))^2)
γ(ky::Float64, kx::Float64) = (γc + hopfx(ky, kx)^2 *(γx-γc))/γp


# The blue-shifted LP dispersion is given by $\epsilon\left(k\right)+2n_{p}\left|X(k)\right|^{2}$.
bsenlp(ky::Float64, kx::Float64, np::Float64) = enlp(ky, kx) + 2np*abs2(hopfx(ky, kx))

# exciton-exciton contact interaction in γp*μm^2  
g = 5e-6/γp
# experimental blue-shift, in units of γp
bshift = 0.0012924472535611464/γp
#global const n_p = bshift/2abs2(hopfx(kpy, kpx))
global const n_p = 43.
# alternatively
#global const n_p = g*abs2(hopfx(kpy, kpx))*650


# $$n_{p}=g\left|X_{p}\right|^{2}\left|\psi_{p}^{\text{ss}}\right|^{2}$$

# inferred exp pump polariton density, in μm^-2
n_p/(g*abs2(hopfx(kpy, kpx)))


# Computational parameters
global const kx = linspace(-12.8, 12.75, 512);


blp = Float64[bsenlp(0., k, n_p) for k in kx];
gma = Float64[γ(0., k) for k in kx]/2;


fig, ax = plt.subplots(figsize=(5, 3))

ax[:plot](kx, blp, "k-.")
ax[:fill_between](kx, blp - gma/2, blp + gma/2, alpha=0.5)

ax[:scatter](kpx, ωp)
    
ax[:set_title]("blue-shifted LP dispersion")
ax[:set_xlim](-8, 8)
ax[:set_xlabel](L"$k_x[\mu m^{-1}]$")
ax[:set_ylabel](L"$\epsilon-\omega_X[\gamma_p]$")


# $$M(k)=\epsilon(k_{p}+k)-\omega_{p}-\frac{i}{2}\gamma(k_{p}+k)+2n_{p}\left|X(k_{p}+k)\right|^{2}$$
# $$Q(k)=n_{p}X^{*}(k_{p}+k)X^{*}(k_{p}-k)$$
# $$R(k) = \frac{C(k_p)}{X(k_p)}C(k+k_p)$$

M(ky::Float64, kx::Float64, np::Float64) = enlp(kpy+ky, kpx+kx) - ωp - im*γ(kpy+ky, kpx+kx)/2 + 2np*abs2(hopfx(kpy+ky, kpx+kx))
Q(ky::Float64, kx::Float64, np::Float64) = np*conj(hopfx(kpy+ky, kpx+kx))*conj(hopfx(kpy-ky, kpx-kx))
R(ky::Float64, kx::Float64) = hopfc(kpy, kpx)/hopfx(kpy, kpx)*hopfc(kpy+ky, kpx+kx)


# $$L(k)=\left(\begin{matrix}M\left(k\right) & Q\left(k\right)e^{2i\phi_{p}}\\
# -e^{-2i\phi_{p}}Q^{*}\left(-k\right) & -M^{*}\left(-k\right)
# \end{matrix}\right)$$

#L(ky::Float64, kx::Float64, np::Float64, φp::Float64) = [M(ky, kx, np) Q(ky, kx, np)*exp(2im*φp); -exp(-2im*φp)*conj(Q(-ky, -kx, np)) -conj(M(-ky, -kx, np))]


# $$w(k)=M\left(k\right)-M^{*}\left(-k\right)$$
# $$z(k)=\left[M\left(k\right)+M^{*}\left(-k\right)\right]^{2}-4Q\left(k\right)Q^{*}\left(-k\right)$$

w(ky::Float64, kx::Float64, np::Float64) = M(ky, kx, np) - conj(M(-ky, -kx, np))
z(ky::Float64, kx::Float64, np::Float64) = (M(ky, kx, np) + conj(M(-ky, -kx, np)))^2 - 4Q(ky, kx, np)*conj(Q(-ky, -kx, np))


# $$\lambda\left(k\right)_{1,2}=\frac{1}{2}w\pm\frac{1}{2}\sqrt{z}$$

λ1(ky::Float64, kx::Float64, np::Float64) = 1/2*w(ky, kx, np) + 1/2*sqrt(z(ky, kx, np))
λ2(ky::Float64, kx::Float64, np::Float64) = 1/2*w(ky, kx, np) - 1/2*sqrt(z(ky, kx, np))


np = n_p
fig, axes = plt.subplots(2,1, figsize=(8,6))
for i = 1:2
    axes[1][:scatter](kx, real([λ1(0., k, np) for k in kx]), s=15, alpha=0.4, color="orange")
    axes[1][:scatter](kx, real([λ2(0., k, np) for k in kx]), s=15, alpha=0.2)
    axes[2][:scatter](kx, imag([λ1(0., k, np) for k in kx]), s=15, alpha=0.4, color="orange")
    axes[2][:scatter](kx, imag([λ2(0., k, np) for k in kx]), s=15, alpha=0.2)
end
for ax in axes
    ax[:set_xlim](kx[1], kx[end])
    ax[:grid]()
end
axes[1][:set_ylabel](L"$\Re(\omega)[\gamma_p]$")
axes[2][:set_ylabel](L"$\Im(\omega)[\gamma_p]$")
axes[2][:set_xlabel](L"$k_x[\mu m^{-1}]$")


# $$\left|X_{p}\right|^{4}n_{p}^{3}+2\left|X_{p}\right|^{2}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}-\left|X_{p}\right|^{4}I_{p}=0$$

function mfroots(kpy::Float64, kpx::Float64, ωp::Float64, ip::Float64)
    xp = hopfx(kpy, kpx)
    ep = enlp(kpy, kpx)
    a = abs2(xp)^2
    b = 2abs2(xp)*(ep - ωp)
    c = 1/4 + (ep - ωp)^2
    d = -a*ip
    r = roots(Poly([a, b, c, d]))
    filter!(x -> isapprox(imag(x), 0.), r)
    rr = real(r)
    shade = Array(ASCIIString, size(rr))
    fill!(shade, "blue")
    for idx = 1:length(rr)
        np = rr[idx]
        for kx = -5:0.05:5
            if imag(λ1(0., kx, np)) > 0 || imag(λ2(0., kx, np)) > 0
                shade[idx] = "red"
                break
            end
        end
    end
    ips = Array(Float64, size(rr))
    fill!(ips, ip)
    return (ips, rr, shade)
end


# $$n_{p}^{3}+\frac{2}{\left|X_{p}\right|^{2}}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\frac{1}{\left|X_{p}\right|^{4}}\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}=I_{p}$$

function findpump(kpy::Float64, kpx::Float64, ωp::Float64, np::Float64)
    xp = hopfx(kpy, kpx)
    ep = enlp(kpy, kpx)
    b = 2/abs2(xp)*(ep-ωp)
    c = 1/abs2(xp)^2*(1/4 + (ep-ωp)^2)
    np^3 + b*np^2 + c*np
end


global const ipchosen = findpump(kpy, kpx, ωp, n_p)


fig, ax = plt.subplots(figsize=(8, 3))
ipmax=2ipchosen
for ip in linspace(0, ipmax, 100)
    ips, nps, colors = mfroots(kpy, kpx, ωp, ip)
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

# nss    -->   np
#325:975 --> 40:59.64

#check
mfroots(kpy, kpx, ωp, ipchosen)


# $$V_d(k) = g_V e^{-k^2 \sigma^2}$$
Vd(ky, kx, σ, gv) = gv*exp(-σ^2*(kx^2+ky^2))

vd = [Vd(ky, kx, 10, 0.5) for kx=-8:0.05:8, ky=-8:0.05:8];


plt.imshow(vd, ColorMap("gray"), origin="lower",
                      extent=[-8, 8, -8, 8])
plt.axis("image")
plt.title("defect potential")
plt.ylim(-8, 8)
plt.xlim(-8, 8)
plt.xlabel(L"$k_x [\mu m^{-1}]$")
plt.ylabel(L"$k_y [\mu m^{-1}]$")




# $$\left|\widetilde{\psi}\left(k+k_{p}\right)\right|^{2}=\frac{n_{p}}{g}\left|\frac{\delta(k)}{X_{p}}+V_{d}\left(k\right)\frac{Q\left(k\right)R^{*}\left(-k\right)-M^{*}\left(-k\right)R\left(k\right)}{M\left(k\right)M^{*}\left(-k\right)-Q\left(k\right)Q^{*}\left(-k\right)}\right|^{2}$$

ψtmom(ky, kx, np, σ, gv) = Vd(ky, kx, σ, gv)*(Q(ky, kx, np)*conj(R(-ky, -kx)) - conj(M(-ky, -kx, np))*R(ky, kx))/(M(ky, kx, np)*conj(M(-ky, -kx, np)) - Q(ky, kx, np)*conj(Q(-ky, -kx, np)))

ψmom = Complex{Float64}[ψtmom(momy, momx, n_p, 0.2, 0.5) for momx in kx, momy in kx];


#add constant part to wavefunction at k = 0
ψmom[257, 257] += 512/hopfx(kpy, kpx);


fig_mom_P, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[1][:imshow](log10(n_p*abs2(ψmom)),
ColorMap("gray"), origin=None, extent=[kx[1], kx[end], kx[1], kx[end]])
axes[2][:plot](kx, log10(n_p*abs2(ψmom[257, :]))) 
axes[2][:grid]()
axes[1][:set_ylabel](L"$k_y [\mu m^{-1}]$")
axes[1][:set_xlabel](L"$k_x [\mu m^{-1}]$")
axes[2][:set_xlabel](L"$k_x [\mu m^{-1}]$")


# $$I(r)=\frac{\vert\psi(r)\vert^{2}}{\vert\psi_p^{\text{ss}}\vert^{2}}=\left|X_{p}\right|^{2}\left|\sum_{k}\left[\frac{\delta(k)}{X_{p}}+V_{d}\left(k\right)\frac{Q\left(k\right)R^{*}\left(-k\right)-M^{*}\left(-k\right)R\left(k\right)}{M\left(k\right)M^{*}\left(-k\right)-Q\left(k\right)Q^{*}\left(-k\right)}\right]e^{ikr}\right|^{2}$$

ψr = abs2(hopfx(kpy, kpx)*fftshift(ifft(512ψmom)));

fig_real_P, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[1][:imshow](ψr, ColorMap("gray"),
                 origin=None)
axes[2][:plot](ψr[257, :])
axes[2][:grid]()
#axes[1][:set_ylabel](y_label_i)
#axes[1][:set_xlabel](x_label_i)
#axes[2][:set_xlabel](x_label_i)
