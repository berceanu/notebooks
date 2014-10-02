using PyPlot
using Contour

# my modules
using OnePump
using Various

const nx = 512
const ny = 512;


const lx = 100
const ly = 100
const sbox = sysbox{Float64}(sider(;l=ly, n=ny), sider(;l=lx, n=nx), sidek(;l=ly, n=ny), sidek(;l=lx, n=nx));


lp = Float64[enlp(0., momx) for momx in sbox.kx]
gma = Float64[γ(0., momx) for momx in sbox.kx]/2;

fig, ax = plt.subplots(figsize=(8, 3))

ax[:plot](sbox.kx, lp, "k-.")
ax[:fill_between](sbox.kx, lp - gma/2, lp + gma/2, alpha=0.5)

ax[:scatter](kpx, ωpγ)
ax[:axhline](y=enlp(kpy, kpx) + 0.5 * sqrt(3), color="r", ls="dashed")

ax[:set_title]("bare LP dispersion")
ax[:set_xlim](-5, 5)
ax[:set_xlabel](L"$k_x[\mu m^{-1}]$")
ax[:set_ylabel](L"$\epsilon-\omega_X[\gamma_p]$")


function scatter(axes, axno, reim::Function, λ::Function, col; ωp=ωpγ, np=ρp)
    axes[axno][:scatter](sbox.kx, reim([λ(0., momx; ωp=ωp, np=np) for momx in sbox.kx]), s=15, alpha=0.6, color=col)
end

fig, axes = plt.subplots(2,1, figsize=(8,6))

scatter(axes, 1, real, λ1, "orange")
scatter(axes, 1, real, λ2, "blue")
scatter(axes, 2, imag, λ1, "orange")
scatter(axes, 2, imag, λ2, "blue")
axes[1][:set_ylim](-15,15)

for ax in axes
    ax[:set_xlim](-5,5)
    ax[:grid]()
end

axes[1][:set_ylabel](L"$\Re(\omega)[\gamma_p]$")
axes[2][:set_ylabel](L"$\Im(\omega)[\gamma_p]$")
axes[2][:set_xlabel](L"$k_x[\mu m^{-1}]$")

#ipchosen = 0.25*ρp/abs2(xp)^2;
ipchosen = 6.5;


fig, ax = plt.subplots(figsize=(8, 3))
ipmax=16
for ip in linspace(0, ipmax, 100)
    ips, nps, colors = mfroots(kpy, kpx; ωp=ωpγ, ip=ip)
    for idx = 1:length(ips)
        ax[:scatter](ips[idx], nps[idx], s=15, alpha=0.4, color=colors[idx])
    end
end
ax[:axvline](x=ipchosen, color="black", ls="dashed")
ax[:grid]()
ax[:set_ylim](0, 1.8)
ax[:set_xlim](0, ipmax)
ax[:set_xlabel](L"$I_p [\gamma_p^3]$")
ax[:set_ylabel](L"$n_p [\gamma_p]$")


velo = -7.5:0.01:-5
dragvel = [drag(sbox; V=[0., vx]) for vx in velo];


fig, ax = plt.subplots(figsize=(20, 2))
ax[:scatter](velo, dragvel)
ax[:grid]()
ax[:set_xlim](-7.5, -5)
ax[:set_xlabel](L"$V$ [arb. u]")
ax[:set_ylabel](L"$\left(F_{d}\right)_{x}$ [arb. u]")


vel = [0., 10.]
ωV = [-vel[2]*momx for momx in sbox.kx];
drag(sbox; V=vel)


# calculate response function
χmat = Complex{Float64}[χ(momy, momx; np=ρp, V=vel) for momy in sbox.ky, momx in sbox.kx];


# crop
cχmat, ckx, cky = cropmat(χmat, sbox.kx, sbox.ky; xcenter=0., ycenter=0., side=8.);

# calculate density modulation in k-space
δρmat = Complex{Float64}[δρ(momy, momx; gV=1., V=vel) for momy in sbox.ky, momx in sbox.kx];


# ifft to obtain density modulation in real-space
δρtmat = real(fftshift(ifft(fftshift(sqrt(nx*ny)*δρmat))));


# crop
cδρtmat, cx, cy = cropmat(δρtmat, sbox.x, sbox.y; xcenter=0., ycenter=0., side=30.);

fig, axes = plt.subplots(4,1, figsize=(6,14))

scatter(axes, 1, real, λ1, "orange")
scatter(axes, 1, real, λ2, "blue")
axes[1][:plot](sbox.kx, ωV, "k-")
axes[1][:set_ylim](-15, 15)

axes[1][:set_xlim](-5,5)
axes[1][:grid]()

axes[1][:set_ylabel](L"$\Re(\omega)[\gamma_p]$")

axes[2][:imshow](log10(abs(cχmat)), ColorMap("gray"),
                 extent=[ckx[1], ckx[end], cky[1], cky[end]])
axes[2][:set_title](L"$\vert\chi(k, -kV)\vert$")
axes[2][:set_ylabel](L"$k_y [\mu m^{-1}]$")
axes[2][:set_xlabel](L"$k_x [\mu m^{-1}]$")

axes[3][:imshow](cδρtmat, ColorMap("gray"), extent=[cx[1], cx[end], cy[1], cy[end]])

axes[4][:plot](cx, vec(cδρtmat[div(size(cδρtmat)[1],2), :]))
axes[4][:set_xlim](cx[1], cx[end])
axes[4][:grid]()

axes[3][:set_ylabel](L"$y [\mu m]$")
axes[3][:set_xlabel](L"$x [\mu m]$")
axes[4][:set_ylabel](L"$I$")
axes[4][:set_xlabel](L"$x [\mu m]$")
