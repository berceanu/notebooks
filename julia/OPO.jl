module OPO

using JSON
using Polynomial

export γp, ωpγ, ρp, hopfx, kpx, kpy, enlp, γ, ωx, λ1, λ2, findpump, mfroots, vdt, ψtmom, gv, xp, χ, δρ, fdt, drag, D, olddrag

# read system parameters from file into dict
#energies in eV
pm = JSON.parsefile("/home/berceanu/notebooks/julia/opoparams.json")

# declare global constants for all values in param file
for (k, v) in pm
    include_string("const " * k * " = $v")
end

const γp = γc + (1/sqrt(1+(Ωr/((1/2*((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))+ωx)-1/2*sqrt(((ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))-ωx)^2+4Ωr^2))
                - (ωc*sqrt(1+(sqrt(kpx^2+kpy^2)/kz)^2))))^2))^2*(γx-γc)

const ωpγ = (ωpev-ωx)/γp
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

bsenlp(ky::Float64, kx::Float64, np::Float64) = enlp(ky, kx) + 2np*abs2(hopfx(ky, kx))

function findpump(kpy::Float64, kpx::Float64; ωp=ωpγ, np=ρp)
    xp = hopfx(kpy, kpx)
    ep = enlp(kpy, kpx)
    b = 2/abs2(xp)*(ep-ωp)
    c = 1/abs2(xp)^2*(1/4 + (ep-ωp)^2)
    np^3 + b*np^2 + c*np
end


M(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp) = enlp(kpy+ky, kpx+kx) - ωp - im*γ(kpy+ky, kpx+kx)/2 + 2np*abs2(hopfx(kpy+ky, kpx+kx))
Q(ky::Float64, kx::Float64; np=ρp) = np*conj(hopfx(kpy+ky, kpx+kx))*conj(hopfx(kpy-ky, kpx-kx))
R(ky::Float64, kx::Float64) = cp/xp*hopfc(kpy+ky, kpx+kx)

# gaussian potential in real space
fdt(y::Float64, x::Float64; σ=sigma, gV=gv) = gV/(2pi*σ^2)*exp(-1/2σ^2*(x^2+y^2))

# gaussian potential in mom space
fd(qy::Float64, qx::Float64; σ=sigma, gV=gv) = gV*exp(-σ^2/2*(qx^2 + qy^2))
vdt(ky::Float64, kx::Float64; σ=sigma, gV=gv, y0=0., x0=0., a=1., b=1., α=0.) = gV*exp(-im*(kx*x0+ky*y0))*exp(-σ^2/4*(a^2+b^2)*(kx^2+ky^2))*exp(-σ^2/4*(a^2-b^2)*(2sin(2α)*kx*ky+cos(2α)*(kx^2 - ky^2)))

# momentum space density perturbation
δρ(qy::Float64, qx::Float64; ωp=ωpγ, np=ρp, V=[0., 0.], gV=gv, σ=sigma) = χ(qy, qx; ωp=ωp, np=np, V=V)*fd(qy, qx; σ=σ, gV=gV)


ψtmom(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp, σ=sigma, gV=gv, y0=0., x0=0., a=1., b=1., α=0.) = (Q(ky, kx; np=np)*conj(R(-ky, -kx))*conj(vdt(-ky, -kx; σ=σ, gV=gV, y0=y0, x0=x0, a=a, b=b, α=α)) - conj(M(-ky, -kx; ωp=ωp, np=np))*R(ky, kx)*vdt(ky, kx; σ=σ, gV=gV, y0=y0, x0=x0, a=a, b=b, α=α))/(M(ky, kx; ωp=ωp, np=np)*conj(M(-ky, -kx; ωp=ωp, np=np)) - Q(ky, kx; np=np)*conj(Q(-ky, -kx; np=np))) 


w(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp) = M(ky, kx; ωp=ωp, np=np) - conj(M(-ky, -kx; ωp=ωp, np=np))
z(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp) = (M(ky, kx; ωp=ωp, np=np) + conj(M(-ky, -kx; ωp=ωp, np=np)))^2 - 4Q(ky, kx; np=np)*conj(Q(-ky, -kx; np=np))



λ1(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp) = 1/2*w(ky, kx; ωp=ωp, np=np) + 1/2*sqrt(z(ky, kx; ωp=ωp, np=np))
λ2(ky::Float64, kx::Float64; ωp=ωpγ, np=ρp) = 1/2*w(ky, kx; ωp=ωp, np=np) - 1/2*sqrt(z(ky, kx; ωp=ωp, np=np))

function mfroots(kpy::Float64, kpx::Float64; ωp=ωpγ, ip=100.)
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
        for momx = -0.75:0.05:0.75
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

#const ρp = mfroots(kpy, kpx; ip=6.5)[2][1] # pump density in units of γp

# determinant of L(q,ω)
function D(qy::Float64, qx::Float64; ωp=ωpγ, np=ρp, V=[0., 0.])
	q = [qy, qx]
	ω = -dot(V, q)
	-(ω - λ1(qy, qx; ωp=ωp, np=np))*(ω - λ2(qy, qx; ωp=ωp, np=np))
end

χnum(qy::Float64, qx::Float64; ωp=ωpγ, np=ρp, ω=0.) = xp * ((conj(M(-qy, -qx; ωp=ωp, np=np)) + ω) * R(qy, qx) - Q(qy, qx; np=np) * conj(R(-qy, -qx)))

# response function
function χ(qy::Float64, qx::Float64; ωp=ωpγ, np=ρp, V=[0., 0.])
	q = [qy, qx]
	freq = -dot(V, q) 
	num = χnum(qy, qx; ωp=ωp, np=np, ω=freq)
	numstar = conj(χnum(-qy, -qx; ωp=ωp, np=np, ω=-freq))
	- (num + numstar)/ D(qy, qx; ωp=ωp, np=np, V=V)
end

function olddrag(box; ωp=ωpγ, np=ρp, V=[0., 0.], gV=1., σ=sigma)

    δρmat = Array(Complex{Float64}, length(box.y), length(box.x)) 
    fdtm = Array(Float64, length(box.y), length(box.x)) 
    norm = sqrt(length(box.x)*length(box.y))
    for j=1:length(box.x), i=1:length(box.y)
	    # calculate density modulation in k-space
	    δρmat[i,j] = norm*δρ(box.ky[i], box.kx[j]; ωp=ωp, np=np, V=V, gV=gV, σ=σ)
	    # calculate r.V on grid
	    fdtm[i,j] = box.x[j]*fdt(box.y[i], box.x[j]; σ=σ, gV=gV)
    end

    # ifft to obtain density modulation in real-space
    δρtmat = real(fftshift(ifft(fftshift(δρmat))))

    return -sum(fdtm.*δρtmat)

end

# drag force
function drag(box; ωp=ωpγ, np=ρp, V=[0., 0.], gV=1., σ=sigma)

    δρmat = Array(Complex{Float64}, length(box.y), length(box.x)) 
    fdtm = Array(Float64, length(box.y), length(box.x)) 
    norm = sqrt(length(box.x)*length(box.y))
    full = length(box.y)
    half = div(full, 2)
    for j=1:length(box.x), i=1:half
	    # calculate density modulation in k-space
	    δρmat[i,j] = norm*δρ(box.ky[i], box.kx[j]; ωp=ωp, np=np, V=V, gV=gV, σ=σ)
	    # calculate r.V on grid
	    fdtm[i,j] = box.x[j]*fdt(box.y[i], box.x[j]; σ=σ, gV=gV)
	    # calculate the other half by symmetry
	    δρmat[full+1-i, j] = δρmat[i,j]
	    fdtm[full+1-i, j] = fdtm[i,j] 
    end

    # ifft to obtain density modulation in real-space
    δρtmat = real(fftshift(ifft(fftshift(δρmat))))

    return -sum(fdtm.*δρtmat)

end


end
