module BerryPhase

using JSON

#export shit..

pm = JSON.parsefile("/home/berceanu/notebooks/julia/berry.json")

const ω0 = -2.95 

#function definitions

getm(i::Int64; N=pm["N"]) = div(i-1,N)-div(N-1,2)
getn(i::Int64; N=pm["N"]) = div(N-1,2)-rem(i-1,N)

#$$a_{i-N}+e^{-i2\pi\alpha m}a_{i-1}+\left[\omega_{0}+i\gamma-\frac{1}{2}\kappa(m^{2}+n^{2})\right]a_{i}+e^{i2\pi\alpha m}a_{i+1}+a_{i+N}=fe^{i\phi_{i}}$$

function genspmat(ω0::Float64; N=pm["N"], α=pm["α"], γ=pm["γ"], κ=pm["κ"])
    if iseven(N)
	    error("even N not allowed")
    end
    I = Int64[]
    J = Int64[]
    V = Complex{Float64}[]
    for i in 1:N^2
        m = getm(i; N=N)
        n = getn(i; N=N)
	list=Int64[]
	push!(list,i)
	push!(V, ω0 + im*γ - 1/2*κ*(m^2+n^2))
        if i > N
	    push!(list,i-N)
	    push!(V, 1)
	    push!(list,i-1)
	    push!(V, exp(-im*2π*α*m))
        elseif i > 1
	    push!(list,i-1)
	    push!(V, exp(-im*2π*α*m))
        end
	if i <= (N-1)*N
            push!(list,i+1)
	    push!(V, exp(im*2π*α*m))
	    push!(list,i+N)
	    push!(V, 1)
        elseif i <= (N-1)*(N+1)
            push!(list,i+1)
	    push!(V, exp(im*2π*α*m))
	end
	append!(J,list)
	append!(I, i .* ones(Int64,length(list)))
    end
    return sparse(I,J,V)
end


function setpump(;f=pm["f"], N=pm["N"])
	# generate matrix of random phases in interval [0,2π)
	ϕ = 2π .* rand(N, N)
	f .* exp(im .* ϕ)
end


function getas(S::SparseMatrixCSC{Complex{Float64},Int64}, fmat::Matrix{Complex{Float64}})
	N = size(fmat)[1]
	fvec = reshape(fmat, N^2)
	reshape(S\fvec, N, N)
end

function calcintensity()
end

function plotintensity()
end

end
