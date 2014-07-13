module OnePumpQuad

export ε, λ1, λ2, ψtmom

ε(ky::Float64, kx::Float64; Δp=0.) = (kx^2 + ky^2)/2 - Δp

λ1(ky::Float64, kx::Float64; vp=(0., 1.9), Δp=0., κ=1.1) = (ky*vp[1] + kx*vp[2]) -im*κ + sqrt(ε(ky, kx; Δp=Δp)*(ε(ky, kx; Δp=Δp) + 2) + 0.*im)
λ2(ky::Float64, kx::Float64; vp=(0., 1.9), Δp=0., κ=1.1) = (ky*vp[1] + kx*vp[2]) -im*κ - sqrt(ε(ky, kx; Δp=Δp)*(ε(ky, kx; Δp=Δp) + 2) + 0.*im)

ψtmom(ky::Float64, kx::Float64; gv=1e-2, vp=(0., 1.9), Δp=0., κ=1.1) = gv*(ε(ky, kx; Δp=Δp) - (ky*vp[1] + kx*vp[2]) + im*κ)/(λ1(ky, kx; Δp=Δp, vp=vp, κ=κ)*λ2(ky, kx; Δp=Δp, vp=vp, κ=κ))

end
