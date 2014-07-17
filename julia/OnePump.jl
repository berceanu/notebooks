module OnePump

using JSON
using Polynomial

export γp, hopfx, kpx, kpy, enlp, γ, ωx, λ1, λ2, findpump, mfroots, vdt, ψtmom, fftfreq, gv, mlp, xp, χ

# read system parameters from file into dict
#energies in eV
pm = JSON.parsefile("/home/berceanu/notebooks/julia/april/params.json")

# declare global constants for all values in param file
for (k, v) in pm
    include_string("const " * k * " = $v")
end

const γp = γc + (1/sqrt(1+(Ωr/((1/2*((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))+ωx)-1/2*sqrt(((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))-ωx)^2+4Ωr^2))
                - (ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))))^2))^2*(γx-γc)

#const ωpev = 1.48283
#const ωp = (ωpev-ωx)/γp

# effective photon mass
const mc = kz^2/(ωc/γp)

# effective LP mass
const mlp = 2mc*(1 - ((ωc - ωx)/γp)/(sqrt(((ωc - ωx)/γp)^2 + 4(Ωr/γp)^2)))^-1.

# We measure energies in units of $\gamma_p$, with the origin set to $\omega_X$.
enc(ky::Float64, kx::Float64) = (ωc * sqrt(1 + (sqrt(kx^2 + ky^2)/kz)^2) - ωx)/γp
enlp(ky::Float64, kx::Float64) = 1/2*enc(ky, kx) - 1/2*sqrt(enc(ky, kx)^2 + 4Ωr^2/γp^2)
hopfx(ky::Float64, kx::Float64) = 1/sqrt(1+((Ωr/γp) / (enlp(ky, kx) - enc(ky, kx)))^2)
hopfc(ky::Float64, kx::Float64) = -1/sqrt(1+((enlp(ky, kx) - enc(ky, kx)) /(Ωr/γp))^2)
γ(ky::Float64, kx::Float64) = (γc + hopfx(ky, kx)^2 *(γx-γc))/γp

const xp = hopfx(kpy, kpx)
const cp = hopfc(kpy, kpx)


# The blue-shifted LP dispersion is given by $\epsilon\left(k\right)+2n_{p}\left|X(k)\right|^{2}$.
bsenlp(ky::Float64, kx::Float64, np::Float64) = enlp(ky, kx) + 2np*abs2(hopfx(ky, kx))

# $$n_{p}^{3}+\frac{2}{\left|X_{p}\right|^{2}}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\frac{1}{\left|X_{p}\right|^{4}}\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}=I_{p}$$

function findpump(kpy::Float64, kpx::Float64; ωp=-30., np=20.)
    xp = hopfx(kpy, kpx)
    ep = enlp(kpy, kpx)
    b = 2/abs2(xp)*(ep-ωp)
    c = 1/abs2(xp)^2*(1/4 + (ep-ωp)^2)
    np^3 + b*np^2 + c*np
end

# $$M(k)=\epsilon(k_{p}+k)-\omega_{p}-\frac{i}{2}\gamma(k_{p}+k)+2n_{p}\left|X(k_{p}+k)\right|^{2}$$
# $$Q(k)=n_{p}X^{*}(k_{p}+k)X^{*}(k_{p}-k)$$
# $$R(k) = \frac{C(k_p)}{X(k_p)}C(k+k_p)$$

M(ky::Float64, kx::Float64; ωp=-30., np=20.) = enlp(kpy+ky, kpx+kx) - ωp - im*γ(kpy+ky, kpx+kx)/2 + 2np*abs2(hopfx(kpy+ky, kpx+kx))
Q(ky::Float64, kx::Float64; np=20.) = np*conj(hopfx(kpy+ky, kpx+kx))*conj(hopfx(kpy-ky, kpx-kx))
R(ky::Float64, kx::Float64) = cp/xp*hopfc(kpy+ky, kpx+kx)

# gaussian potential in real space
fdt(y::Float64, x::Float64; σ=1., gV=gv) = gV/(2pi*σ^2)*exp(-1/2σ^2*(x^2+y^2))

# gaussian potential in mom space
fd(qy::Float64, qx::Float64; σ=1., gV=gv) = gV*exp(-σ^2/2*(qx^2 + qy^2))
vdt(ky::Float64, kx::Float64; σ=1., gV=gv, y0=0., x0=0., a=1., b=1., α=0.) = gV*exp(-im*(kx*x0+ky*y0))*exp(-σ^2/4*(a^2+b^2)*(kx^2+ky^2))*exp(-σ^2/4*(a^2-b^2)*(2sin(2α)*kx*ky+cos(2α)*(kx^2 - ky^2)))

# momentum space density perturbation
function δρ(qy::Float64, qx::Float64; ωp=-30., np=20., V=[0., 0.], gV=gv, σ=1.)
	q = [qy, qx]
	ω = -dot(V, q) 
	χ(qy, qx; ωp=ωp, np=np, V=V)*fd(qy, qx; σ=σ, gV=gV)
end

ψtmom(ky::Float64, kx::Float64; ωp=-30., np=20., σ=1., gV=gv, y0=0., x0=0., a=1., b=1., α=0.) = (Q(ky, kx; np=np)*conj(R(-ky, -kx))*conj(vdt(-ky, -kx; σ=σ, gV=gV, y0=y0, x0=x0, a=a, b=b, α=α)) - conj(M(-ky, -kx; ωp=ωp, np=np))*R(ky, kx)*vdt(ky, kx; σ=σ, gV=gV, y0=y0, x0=x0, a=a, b=b, α=α))/(M(ky, kx; ωp=ωp, np=np)*conj(M(-ky, -kx; ωp=ωp, np=np)) - Q(ky, kx; np=np)*conj(Q(-ky, -kx; np=np))) 

# $$w(k)=M\left(k\right)-M^{*}\left(-k\right)$$
# $$z(k)=\left[M\left(k\right)+M^{*}\left(-k\right)\right]^{2}-4Q\left(k\right)Q^{*}\left(-k\right)$$

w(ky::Float64, kx::Float64; ωp=-30., np=20.) = M(ky, kx; ωp=ωp, np=np) - conj(M(-ky, -kx; ωp=ωp, np=np))
z(ky::Float64, kx::Float64; ωp=-30., np=20.) = (M(ky, kx; ωp=ωp, np=np) + conj(M(-ky, -kx; ωp=ωp, np=np)))^2 - 4Q(ky, kx; np=np)*conj(Q(-ky, -kx; np=np))


# $$\lambda\left(k\right)_{1,2}=\frac{1}{2}w\pm\frac{1}{2}\sqrt{z}$$

λ1(ky::Float64, kx::Float64; ωp=-30., np=20.) = 1/2*w(ky, kx; ωp=ωp, np=np) + 1/2*sqrt(z(ky, kx; ωp=ωp, np=np))
λ2(ky::Float64, kx::Float64; ωp=-30., np=20.) = 1/2*w(ky, kx; ωp=ωp, np=np) - 1/2*sqrt(z(ky, kx; ωp=ωp, np=np))

# determinant of L(q,ω)
function D(qy::Float64, qx::Float64; ωp=-30., np=20., V=[0., 0.])
	q = [qy, qx]
	ω = -dot(V, q)
	-(ω - λ1(qy, qx; ωp=ωp, np=np))*(ω - λ2(qy, qx; ωp=ωp, np=np))
end

# response function
# TODO: express num2 as num1(-q)^*
function χ(qy::Float64, qx::Float64; ωp=-30., np=20., V=[0., 0.])
	q = [qy, qx]
	ω = -dot(V, q) 
	num1 = xp*((conj(M(-qy, -qx; ωp=ωp, np=np)) + ω)*R(qy, qx) - Q(qy, qx; np=np)*conj(R(-qy, -qx)))
	num2 = conj(xp)*((M(qy, qx; ωp=ωp, np=np)-ω)*conj(R(-qy, -qx)) - conj(Q(-qy, -qx; np=np))*R(qy, qx))
	numerator = num1 + num2
	- numerator / D(qy, qx; ωp=ωp, np=np, V=V)
end


# $$\left|X_{p}\right|^{4}n_{p}^{3}+2\left|X_{p}\right|^{2}\left(\epsilon_{p}-\omega_{p}\right)n_{p}^{2}+\left[\frac{1}{4}+\left(\epsilon_{p}-\omega_{p}\right)^{2}\right]n_{p}-\left|X_{p}\right|^{4}I_{p}=0$$

function mfroots(kpy::Float64, kpx::Float64; ωp=-30., ip=100., np=20.)
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
            if imag(λ1(0., momx; ωp=ωp, np=np)) > 0 || imag(λ2(0., momx; ωp=ωp, np=np)) > 0
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
