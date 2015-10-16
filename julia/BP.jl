module BP

using Polynomials
#import PyPlot
#using LaTeXStrings

#golden ratio
ϕgold = 1.618

getm(i::Int64,N::Int64) = div(i-1,N)-div(N-1,2)
getn(i::Int64,N::Int64) = div(N-1,2)-rem(i-1,N)


## function extractpath(l1::Int,l2::Int,c1::Int,c2::Int, M::Matrix{Float64})
##     path=Float64[]
##     append!(path, M[l1:l2,c1])
##     append!(path, vec(M[l2,c1:c2])[2:end])
##     append!(path, M[l2:-1:l1,c2][2:end])
##     append!(path, vec(M[l1,c2:-1:c1])[2:end-1])
##     path
## end

function countnonzeros(N::Int)
    k = N^2 #elements on the diagonal are all nonzero

    # maximum value of m or n indices
    maxm = div(N-1,2)

    for i in 1:N^2
        m = getm(i,N)
        n = getn(i,N)
        if n==maxm && m==-maxm #tl corner
            k+=2
        elseif n==maxm && m==maxm #tr corner
            k+=2
        elseif n==-maxm && m==maxm #br corner
            k+=2
        elseif n==-maxm && m==-maxm #bl corner
            k+=2
        elseif n==maxm #t edge
            k+=3
        elseif m==maxm # r edge
            k+=3
        elseif n==-maxm # b edge
            k+=3
        elseif m==-maxm # l edge
            k+=3
        else # bulk
            k+=4
        end
    end
    return k
end


function genspmat(l::Function,r::Function,u::Function,d::Function,s::Function, N::Int,nz::Int,α::Float64)
    # Preallocate
    I = Array(Int64,nz)
    J = Array(Int64,nz)
    V = Array(Complex{Float64},nz)

    function setnzelem(i::Int,n::Int,m::Int; pos="self")
        if pos=="left"
            k += 1
            J[k] = i-N; I[k] = i; V[k] = l(n,m,α)
        elseif pos=="right"
            k += 1
            J[k] = i+N; I[k] = i; V[k] = r(n,m,α)
        elseif pos=="up"
            k += 1
            J[k] = i-1; I[k] = i; V[k] = u(n,m,α)
        elseif pos=="down"
            k += 1
            J[k] = i+1; I[k] = i; V[k] = d(n,m,α)
        elseif pos=="self"
            k += 1
            J[k] = i; I[k] = i; V[k] = s(n,m,α)
        end
    end
            
    # maximum value of m or n indices
    maxm = div(N-1,2)

    k = 0
    for i in 1:N^2
        m = getm(i,N)
        n = getn(i,N)
        setnzelem(i,n,m; pos="self")
        #corners
        #top left
        if n==maxm && m==-maxm
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="down")
        #top right
        elseif n==maxm && m==maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="down")
        #bottom right
        elseif n==-maxm && m==maxm 
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="up")
        #bottom left
        elseif n==-maxm && m==-maxm 
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="up")
        #edges
        #top
        elseif n == maxm
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="down")
        #right
        elseif m == maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="down")
        #bottom
        elseif n == -maxm
            setnzelem(i,n,m; pos="left")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="right")
        #left
        elseif m == -maxm
            setnzelem(i,n,m; pos="down")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="right")
        else #bulk
            setnzelem(i,n,m; pos="down")
            setnzelem(i,n,m; pos="up")
            setnzelem(i,n,m; pos="right")
            setnzelem(i,n,m; pos="left")
        end
    end
    return sparse(I,J,V)
end


function mbz(data, q, N)
    #l = div(N-1, q)
    l = N

    V = zeros(Float64,N,l)
    
    for i in 1:l
        idx = i
        while idx <= q*N+1
            V[:,i] += data[:,idx]
            idx += l
        end
    end

    V/(4π^2)
end


function momsphhmat(kx0::Float64, ky::Float64, α::Float64,q::Int)
    du = ones(Complex{Float64}, q-1) #upper diagonal
    d = Complex{Float64}[2*cos(kx0 + 2*π*α*j) for j in 1:q] #main diagonal
    mat = full(SymTridiagonal(d, du))
    mat[1,q] = exp(-im*q*ky)
    mat[q,1] = exp(im*q*ky)
    return -mat
end


function hhladder(α::Float64, q::Int)
    kx0 = 0.
    ky = linspace(-π, π, 100)
    E = Array(Float64, 100,q)
    for c in 1:100
        M = Hermitian(momsphhmat(kx0, ky[c], α, q))
        E[c,:] = eigvals(M) #q eigenvalues
    end

    E
end


function compute_hermite_polynomial(n)
    P = Poly([1])
    const x = Poly([0; 1])                                                                                 
    for i = 1:n
        P = 2x*P - polyder(P)
    end
    P
end

function χ(kx0, ky, α, β)
    l = sqrt(2π*α)

    sum = zero(Complex{Float64})
    for j in -20:20 #truncate the sum
        H = polyval(compute_hermite_polynomial(β), kx0/l + j*l)
        sum += exp(-im*ky*j) * exp(-(kx0 + j*l^2)^2/(2l^2)) * H
    end

    sum
end

########################
#various pumping schemes
########################
function δpmp(N::Int; A=1., seed=0, σ=0., n0=0, m0=0)
    i = (m0+div(N-1,2)) * N + (div(N-1,2)-n0) + 1
    f = zeros(Complex{Float64}, N^2)
    f[i] = A * one(Complex{Float64})
    f
end
function gausspmp(N::Int; A=1., seed=0, σ=1., n0=0, m0=0)
    x0=m0 
    y0=n0
    f = zeros(Complex{Float64}, N,N)
    m = [-div(N-1,2):div(N-1,2)]
    n = [div(N-1,2):-1:-div(N-1,2)]
    for c in 1:N, l in 1:N
        x = m[c]
        y = n[l]
        f[l,c] = A*exp(-1/(2σ^2)*((x-x0)^2 + (y-y0)^2))
    end
    reshape(f, N^2)
end
function randpmp(N::Int; A=1., seed=123, σ=0., n0=0, m0=0)
    # seed the RNG #
    srand(seed)
    # generate matrix of random phases in interval [0,2π)
    ϕ = 2π .* rand(N^2)
    A .* exp(im .* ϕ)
end
function homopmp(N::Int; A=1., seed=0, σ=0., n0=0, m0=0)
    A .* ones(Complex{Float64}, N^2)
end


#########################
#arbitrary resolution fft
#########################
function myfft2(ψr::Matrix{Complex{Float64}}, k1::Float64, k2::Float64, xs1::Float64, xs2::Float64, Δx1::Float64, Δx2::Float64)
    (N1,N2) = size(ψr)

    s = zero(Complex{Float64})
    for n2 in 1:N2, n1 in 1:N1
        xn1 = xs1 + (n2-1)*Δx1 #x
        xn2 = xs2 + (n1-1)*Δx2 #y
        cexp = exp(-im*(k1*xn1 + k2*xn2))
        s += ψr[n1,n2]*cexp
    end
    s
end
function myfft2(ψr::Matrix{Complex{Float64}}, k1::FloatRange{Float64}, k2::FloatRange{Float64})
    N1 = length(k2); N2 = length(k1)
    

    out = Array(Complex{Float64}, N1,N2)
    for j in 1:N2, i in 1:N1
        out[i,j] = myfft2(ψr, k1[j], k2[i], 0., 0., 1., 1.)
    end
    out
end


########################################
#publication-quality plots in matplotlib
########################################
function matplotspect(ν::Vector{Float64}, I::Vector{Float64}, spect::Vector{Float64}, ω::Float64; vert=true)
    mx = maximum(I)
    
    fig, ax = PyPlot.plt.subplots(1, 1, figsize=(4ϕgold, 4))

    ax[:plot](ν, I, "k")

    vert && ax[:axvline](x = ω, color="red", ls="dashed")
    ax[:vlines](spect, 0, mx/2, colors="orange", linestyles="dashed")
    
    ax[:set_xlim](ν[1], ν[end])
    ax[:set_ylim](0, mx)

    ax[:set_ylabel](L"$\sum_{m,n} |a_{m,n}|^2$ [a.u.]")
    ax[:set_xlabel](L"$\omega_0 [J]$")

    fig[:savefig]("spectrum", bbox_inches="tight")
    PyPlot.plt.close(fig);
end
function matplotcomp(ky::Vector{Float64}, D::Matrix{Float64}, χ::Vector{Complex{Float64}},ζ::Int)
    l=size(D)[2]
    
    fig, ax = PyPlot.plt.subplots(1, 1, figsize=(4ϕgold, 4))

    ax[:plot](ky, abs2(χ), "r")
    ax[:plot](ky, D[:,div(l-1,2)+1]/ζ, "k")

    ax[:set_xlim](-π, π)

    ax[:set_ylabel](L"$|\chi_{\beta}(p_x^0=0, p_y)|^2$")
    ax[:set_xlabel](L"$p_y$")

    fig[:savefig]("compare", bbox_inches="tight")
    PyPlot.plt.close(fig);
end
function matplot2D(m::Matrix{Float64}, xdata::Vector{Float64}, ydata::Vector{Float64}
    ;ratio=1., xlabel=L"$\kappa$", ylabel=L"$q$", zlabel=L"$\eta_{ZPE}$", figname="sample")

    fig, ax = PyPlot.plt.subplots(figsize=(4, 4))
    img = ax[:imshow](m, origin="upper", PyPlot.ColorMap("hot"), interpolation="none",
    extent=[minimum(xdata), maximum(xdata), minimum(ydata), maximum(ydata)], aspect=ratio)
    
    ax[:set_xlim](minimum(xdata), maximum(xdata))
    ax[:set_ylim](minimum(ydata), maximum(ydata))
    ax[:set_xlabel](xlabel)
    ax[:set_ylabel](ylabel)

    cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
    cbar[:ax][:tick_params](labelsize=7)
    cbar[:set_label](zlabel)
    
    fig[:savefig](figname, bbox_inches="tight")
    PyPlot.plt.close(fig);
end
function saveplots(ψreal,ψrealpmp,ψmom,ψmompmp,ψmbz,ψmbzpmp, x,y,kx,ky,kxmbz, q)
    matplot2D(ψreal,x,y; xlabel=L"$m$", ylabel=L"$n$", zlabel=L"$|a_{m,n}|^2$", figname="real")
    matplot2D(ψrealpmp,x,y; xlabel=L"$m$", ylabel=L"$n$", zlabel=L"$|a_{m,n}|^2$", figname="realpmp")
    matplot2D(ψmom,kx,ky; xlabel=L"$p_x$", ylabel=L"$p_y$", zlabel=L"$|a_{p_x,p_y}|^2$", figname="mom")
    matplot2D(ψmompmp,kx,ky; xlabel=L"$p_x$", ylabel=L"$p_y$", zlabel=L"$|a_{p_x,p_y}|^2$", figname="mompmp")
    matplot2D(ψmbz,kxmbz,ky; ratio=1/q, xlabel=L"$p_x^0$", ylabel=L"$p_y$", zlabel=L"$|a_{p_x^0,p_y}|^2$", figname="mbz")
    matplot2D(ψmbzpmp,kxmbz,ky; ratio=1/q, xlabel=L"$p_x^0$", ylabel=L"$p_y$", zlabel=L"$|a_{p_x^0,p_y}|^2$", figname="mbzpmp")
end

end
