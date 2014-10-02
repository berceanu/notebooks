module OnePumpExp

using JSON
using Polynomial

export γp, ωpγ, ρp, hopfx, kpx, kpy, enlp, γ, ωx, ψtmom, gv, mlp, xp

#"kz": 20.0,
#"ωx": 1.483952,
#"ωc": 1.483952,
#"Ωr": 3e-3,
#"γx": 6.582e-7,
#"γc": 1.316e-4,
#"kpx": 0.89,
#"kpy": 0.0,
#"gv": 0.5


# read system parameters from file into dict
#energies in eV
pm = JSON.parsefile("/home/berceanu/notebooks/julia/april/params.json")

# declare global constants for all values in param file
for (k, v) in pm
    include_string("const " * k * " = $v")
end

const γp = γc + (1/sqrt(1+(Ωr/((1/2*((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))+ωx)-1/2*sqrt(((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))-ωx)^2+4Ωr^2))
                - (ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))))^2))^2*(γx-γc)

const ρp = 0.72714167 # density from pump state in OPO
const sigma = 0.1 #width of gaussian defect



# We measure energies in units of $\gamma_p$, with the origin set to $\omega_X$.
enc(ky::Float64, kx::Float64) = (ωc * sqrt(1 + (sqrt(kx^2 + ky^2)/kz)^2) - ωx)/γp
enlp(ky::Float64, kx::Float64) = 1/2*enc(ky, kx) - 1/2*sqrt(enc(ky, kx)^2 + 4Ωr^2/γp^2)
hopfx(ky::Float64, kx::Float64) = 1/sqrt(1+((Ωr/γp) / (enlp(ky, kx) - enc(ky, kx)))^2)
hopfc(ky::Float64, kx::Float64) = -1/sqrt(1+((enlp(ky, kx) - enc(ky, kx)) /(Ωr/γp))^2)
γ(ky::Float64, kx::Float64) = (γc + hopfx(ky, kx)^2 *(γx-γc))/γp

const xp = hopfx(kpy, kpx)
const cp = hopfc(kpy, kpx)

const δ = -1.1007151313383403 # -1.10...
const ωpγ = enlp(kpy, kpx) + δ



M(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp) = enlp(kpy+ky, kpx+kx) - ωp - im*γ(kpy+ky, kpx+kx)/2 + 2np*abs2(hopfx(kpy+ky, kpx+kx))
Q(ky::Float64, kx::Float64; np=ρp) = np*conj(hopfx(kpy+ky, kpx+kx))*conj(hopfx(kpy-ky, kpx-kx))
R(ky::Float64, kx::Float64) = cp/xp*hopfc(kpy+ky, kpx+kx)


# gaussian potential in mom space
vdt(ky::Float64, kx::Float64; σ=sigma, gV=gv, y0=0., x0=0., a=1., b=1., α=0.) = gV*exp(-im*(kx*x0+ky*y0))*exp(-σ^2/4*(a^2+b^2)*(kx^2+ky^2))*exp(-σ^2/4*(a^2-b^2)*(2sin(2α)*kx*ky+cos(2α)*(kx^2 - ky^2)))



ψtmom(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp, σ=sigma, gV=gv, y0=0., x0=0., a=1., b=1., α=0.) = (Q(ky, kx; np=np)*conj(R(-ky, -kx))*conj(vdt(-ky, -kx; σ=σ, gV=gV, y0=y0, x0=x0, a=a, b=b, α=α)) - conj(M(-ky, -kx; ωp=ωp, np=np))*R(ky, kx)*vdt(ky, kx; σ=σ, gV=gV, y0=y0, x0=x0, a=a, b=b, α=α))/(M(ky, kx; ωp=ωp, np=np)*conj(M(-ky, -kx; ωp=ωp, np=np)) - Q(ky, kx; np=np)*conj(Q(-ky, -kx; np=np))) 

end
