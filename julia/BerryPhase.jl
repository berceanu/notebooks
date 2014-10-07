module BerryPhase

using JSON, PyPlot
#turn off interactive plotting
pygui(false)

export genspmat, getass

pm = JSON.parsefile("/home/berceanu/notebooks/julia/berry.json")

const ω0 = -2.95 

#various plotting parameters#
#golden ratio
ϕgold = 1.618
#bigger fonts
matplotlib["rcParams"][:update](["font.size" => 18, "font.family" => "serif"])

#function definitions#

function setpump(;f=pm["f"], N=pm["N"])
	# generate matrix of random phases in interval [0,2π)
	ϕ = 2π .* rand(N, N)
	f .* exp(im .* ϕ)
end

getm(i::Int64; N=pm["N"]) = div(i-1,N)-div(N-1,2)
getn(i::Int64; N=pm["N"]) = div(N-1,2)-rem(i-1,N)


function countnonzeros(; N=pm["N"])
    k = N^2
    for i in 1:N^2
        if i > N
            k += 2
        elseif i > 1
            k += 1
        end
        if i <= (N-1)*N
            k += 2
        elseif i <= (N-1)*(N+1)
            k += 1
        end
    end
    return k
end

function genspmat(ω0::Float64; N=pm["N"], α=pm["α"], γ=pm["γ"], κ=pm["κ"])
    iseven(N) && error("N must be odd")

    # Determine memory usage
    nz = countnonzeros(; N=N)
    
    # Preallocate
    I = Array(Int64,nz)
    J = Array(Int64,nz)
    V = Array(Complex{Float64},nz)

    k = 0
    for i in 1:N^2
        m = getm(i; N=N)
        n = getn(i; N=N)
        k += 1
        J[k] = i; I[k] = i; V[k] = ω0 + im*γ - 1/2*κ*(m^2+n^2)
        if i > N
	    k += 1
            J[k] = i-N; I[k] = i; V[k] = 1
	    k += 1
            J[k] = i-1; I[k] = i; V[k] = exp(-im*2π*α*m)
        elseif i > 1
	    k += 1
            J[k] = i-1; I[k] = i; V[k] = exp(-im*2π*α*m)
        end
	if i <= (N-1)*N
	    k += 1
            J[k] = i+1; I[k] = i; V[k] = exp(im*2π*α*m)
	    k += 1
            J[k] = i+N; I[k] = i; V[k] = 1
        elseif i <= (N-1)*(N+1)
	    k += 1
            J[k] = i+1; I[k] = i; V[k] = exp(im*2π*α*m)
	end
    end
    return sparse(I,J,V)
end

function getass(S::SparseMatrixCSC, fmat::Matrix)
	N = size(fmat)[1]
	fvec = reshape(fmat, N^2)
	reshape(S\fvec, N, N)
end

function calcintensity(ω0::Float64)
	P = setpump()
	S = genspmat(ω0)
	X = getass(S,P)
	return sum(abs2(X))
end

function plotintensity(x, y)
	fig, ax = plt.subplots(1, 1, figsize=(4ϕgold, 4))

	ax[:plot](x, y, "black")

	ax[:set_xlim](x[1], x[end])
	ax[:set_ylim](0, maximum(y))
	ax[:yaxis][:set_ticks]([0, div(maximum(y),2), maximum(y)])
	#ax[:xaxis][:set_ticks]([,])
	ax[:set_ylabel](L"$\sum_{m,n} |a_{m,n}|^2$ [a.u.]")
	ax[:set_xlabel](L"$\omega_0 [J]$")
	fig[:savefig]("fig_berry_int", bbox_inches="tight")
end

function plotreal(data; N=pm["N"])
	fig, ax = plt.subplots(figsize=(4, 4))

	img = ax[:imshow](data, origin="upper",
				extent=[-div(N-1,2), div(N-1,2), -div(N-1,2), div(N-1,2)])

	ax[:set_ylim](-div(N-1,2), div(N-1,2))
	ax[:set_xlim](-div(N-1,2), div(N-1,2))
	ax[:set_xlabel](L"$m$")
	ax[:set_ylabel](L"$n$")

	ax[:xaxis][:set_ticks]([-60, -30, 0, 30, 60])
	ax[:yaxis][:set_ticks]([-60, -30, 0, 30, 60])

	cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
	cbar[:ax][:tick_params](labelsize=7)

	fig[:savefig]("fig_berry_real", bbox_inches="tight")
end

end
