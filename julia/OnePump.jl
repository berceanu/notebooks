module OnePump

using JSON
using Polynomial

export γp, hopfx, kpx, kpy, bsenlp, γ, ωp, λ1, λ2, findpump, mfroots, vd, ψtmom

# read system parameters from file into dict
#energies in eV
pm = JSON.parsefile("/home/berceanu/notebooks/julia/april/params.json")

# declare global constants for all values in param file
for (k, v) in pm
    include_string("const " * k * " = $v")
end

const γp = γc + (1/sqrt(1+(Ωr/((1/2*((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))+ωx)-1/2*sqrt(((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))-ωx)^2+4Ωr^2))
                - (ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))))^2))^2*(γx-γc)
const ωp = (ωpev-ωx)/γp

# We measure energies in units of $\gamma_p$, with the origin set to $\omega_X$.
enc(ky::Float64, kx::Float64) = (ωc * sqrt(1 + (sqrt(kx^2 + ky^2)/kz)^2) - ωx)/γp
enlp(ky::Float64, kx::Float64) = 1/2*enc(ky, kx) - 1/2*sqrt(enc(ky, kx)^2 + 4Ωr^2/γp^2)
hopfx(ky::Float64, kx::Float64) = 1/sqrt(1+((Ωr/γp) / (enlp(ky, kx) - enc(ky, kx)))^2)
hopfc(ky::Float64, kx::Float64) = -1/sqrt(1+((enlp(ky, kx) - enc(ky, kx)) /(Ωr/γp))^2)
γ(ky::Float64, kx::Float64) = (γc + hopfx(ky, kx)^2 *(γx-γc))/γp

# The blue-shifted LP dispersion is given by $\epsilon\left(k\right)+2n_{p}\left|X(k)\right|^{2}$.
bsenlp(ky::Float64, kx::Float64, np::Float64) = enlp(ky, kx) + 2np*abs2(hopfx(ky, kx))

# $$n_{p}^{3}+\frac{2}{\left|X_{p}\right|^{2}}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\frac{1}{\left|X_{p}\right|^{4}}\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}=I_{p}$$

function findpump(kpy::Float64, kpx::Float64, ωp::Float64, np::Float64)
    xp = hopfx(kpy, kpx)
    ep = enlp(kpy, kpx)
    b = 2/abs2(xp)*(ep-ωp)
    c = 1/abs2(xp)^2*(1/4 + (ep-ωp)^2)
    np^3 + b*np^2 + c*np
end

# $$M(k)=\epsilon(k_{p}+k)-\omega_{p}-\frac{i}{2}\gamma(k_{p}+k)+2n_{p}\left|X(k_{p}+k)\right|^{2}$$
# $$Q(k)=n_{p}X^{*}(k_{p}+k)X^{*}(k_{p}-k)$$
# $$R(k) = \frac{C(k_p)}{X(k_p)}C(k+k_p)$$

M(ky::Float64, kx::Float64, np::Float64) = enlp(kpy+ky, kpx+kx) - ωp - im*γ(kpy+ky, kpx+kx)/2 + 2np*abs2(hopfx(kpy+ky, kpx+kx))
Q(ky::Float64, kx::Float64, np::Float64) = np*conj(hopfx(kpy+ky, kpx+kx))*conj(hopfx(kpy-ky, kpx-kx))
R(ky::Float64, kx::Float64) = hopfc(kpy, kpx)/hopfx(kpy, kpx)*hopfc(kpy+ky, kpx+kx)

vd(ky::Float64, kx::Float64, σ::Float64, gv::Float64, y0::Float64, x0::Float64) = gv/σ*exp(-1/2*σ^2*(kx^2+ky^2))*exp(-im*(x0*kx+y0*ky))

ψtmom(ky::Float64, kx::Float64, np::Float64, σ::Float64, gv::Float64, y0::Float64, x0::Float64) = (Q(ky, kx, np)*conj(R(-ky, -kx))*conj(vd(-ky, -kx, σ, gv, y0, x0)) - conj(M(-ky, -kx, np))*R(ky, kx)*vd(ky, kx, σ, gv, y0, x0))/(M(ky, kx, np)*conj(M(-ky, -kx, np)) - Q(ky, kx, np)*conj(Q(-ky, -kx, np)))

# $$w(k)=M\left(k\right)-M^{*}\left(-k\right)$$
# $$z(k)=\left[M\left(k\right)+M^{*}\left(-k\right)\right]^{2}-4Q\left(k\right)Q^{*}\left(-k\right)$$

w(ky::Float64, kx::Float64, np::Float64) = M(ky, kx, np) - conj(M(-ky, -kx, np))
z(ky::Float64, kx::Float64, np::Float64) = (M(ky, kx, np) + conj(M(-ky, -kx, np)))^2 - 4Q(ky, kx, np)*conj(Q(-ky, -kx, np))


# $$\lambda\left(k\right)_{1,2}=\frac{1}{2}w\pm\frac{1}{2}\sqrt{z}$$

λ1(ky::Float64, kx::Float64, np::Float64) = 1/2*w(ky, kx, np) + 1/2*sqrt(z(ky, kx, np))
λ2(ky::Float64, kx::Float64, np::Float64) = 1/2*w(ky, kx, np) - 1/2*sqrt(z(ky, kx, np))


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
        for momx = -5:0.05:5
            if imag(λ1(0., momx, np)) > 0 || imag(λ2(0., momx, np)) > 0
                shade[idx] = "red"
                break
            end
        end
    end
    ips = Array(Float64, size(rr))
    fill!(ips, ip)
    return (ips, rr, shade)
end


end
