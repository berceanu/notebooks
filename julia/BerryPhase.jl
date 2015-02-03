module BerryPhase

using JSON, PyPlot, DSP

#turn off interactive plotting#
#pygui(false)

export pm, cω0, genspmat, getass, plotintensity, plotreal, calcintensity, plotbz

pm = JSON.parsefile("/home/berceanu/notebooks/julia/berry.json")

const cω0 = -2.961

#various plotting parameters#
#golden ratio
ϕgold = 1.618
#bigger fonts
matplotlib["rcParams"][:update](["font.size" => 18, "font.family" => "serif"])


#function definitions#

function setpump(;f=pm["f"], N=pm["N"], seed=1234, random = 1)
    # seed the RNG #
    srand(seed)
    # generate matrix of random phases in interval [0,2π)
    ϕ = 2π .* rand(N^2) .* random
    f .* exp(im .* ϕ)
end

function setpump(n,m; f=pm["f"], N=pm["N"])
    fv = zeros(Complex{Float64}, N^2)
    i = geti(n,m; N=N)
    fv[i] = f
    return fv
end

getm(i::Int64; N=pm["N"]) = div(i-1,N)-div(N-1,2)
getn(i::Int64; N=pm["N"]) = div(N-1,2)-rem(i-1,N)

function geti(n,m; N=pm["N"])
    d = m+div(N-1,2) #div(i-1,N)
    r = div(N-1,2)-n #rem(i-1,N)
    d*N+r+1
end


function countnonzeros(; N=pm["N"])
    k = N^2 #elements on the diagonal are all nonzero

    # maximum value of m or n indices
    maxm = div(N-1,2)

    for i in 1:N^2
        m = getm(i; N=N)
        n = getn(i; N=N)
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

function mapping(; N=pm["N"])
    for i in 1:N^2
        n = getn(i; N=N)
        m = getm(i; N=N)
        println("i=",i,"; ","(n,m)= ","(",n,",", m,")")
    end
end

function genspmat(ω0::Float64; N=pm["N"], α=pm["α"], γ=pm["γ"], κ=pm["κ"])
    iseven(N) && error("N must be odd")

    # Determine memory usage
    nz = countnonzeros(; N=N)

    # Preallocate
    I = Array(Int64,nz)
    J = Array(Int64,nz)
    V = Array(Complex{Float64},nz)

    function setnzelem(i,n,m; pos="self")
        if pos=="left"
            k += 1
            J[k] = i-N; I[k] = i; V[k] = 1
        elseif pos=="right"
            k += 1
            J[k] = i+N; I[k] = i; V[k] = 1
        elseif pos=="up"
            k += 1
            J[k] = i-1; I[k] = i; V[k] = exp(-im*2π*α*m)
        elseif pos=="down"
            k += 1
            J[k] = i+1; I[k] = i; V[k] = exp(im*2π*α*m)
        elseif pos=="self"
            k += 1
            J[k] = i; I[k] = i; V[k] = ω0 + im*γ - 1/2*κ*(m^2+n^2)
        end
    end
            
    # maximum value of m or n indices
    maxm = div(N-1,2)

    k = 0
    for i in 1:N^2
        m = getm(i; N=N)
        n = getn(i; N=N)
        #self interaction is always present
        setnzelem(i,n,m)
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

function getass(S::SparseMatrixCSC, fvec::Vector; N=pm["N"])
    reshape(S\fvec, N, N)
end

function calcintensity(freq::Range; seed = 1234, random = 1)
    P = setpump(;seed = seed, random = random)
    Float64[sum(abs2(getass(genspmat(ω0), P))) for ω0 in freq]
end

function calcintensity(freq::Range, n, m)
    P = setpump(n,m)
    Float64[sum(abs2(getass(genspmat(ω0), P))) for ω0 in freq]
end

function plotintensity(; N=pm["N"], start = -4., stp = 0.001, stop = -2., seed = 1234, random = 1)
    x = start:stp:stop
    y = calcintensity(x; seed = seed, random = random)./(N^2)
    mx = maximum(y)

    spectrum = filter(x-> x > start && x < stop, theeigs)

    fig, ax = plt.subplots(1, 1, figsize=(4ϕgold, 4))

    ax[:plot](x, y, "k")

    ax[:axvline](x = cω0, color="red", ls="dashed")
    ax[:vlines](spectrum, 0, mx/2, colors="r", linestyles="dashed")
    
    ax[:set_xlim](x[1], x[end])
    ax[:set_ylim](0, mx)
    tks = int(linspace(0, mx, 5))
    ax[:yaxis][:set_ticks](tks)
    ax[:set_ylabel](L"$\sum_{m,n} |a_{m,n}|^2$ [a.u.]")
    ax[:set_xlabel](L"$\omega_0 [J]$")
    fig[:savefig]("fig_berry_int", bbox_inches="tight")
end

function plotintensity(n,m; N=pm["N"], start = 0., stp = 0.001, stop = 2., vert=1.)
    x = start:stp:stop
    y = calcintensity(x,n,m)./(N^2)
    mx = maximum(y)

    spectrum = filter(x-> x > start && x < stop, theeigs)

    fig, ax = plt.subplots(1, 1, figsize=(4ϕgold, 4))

    ax[:plot](x, y, "k")

    ax[:axvline](x = vert, color="red", ls="dashed")
    ax[:vlines](spectrum, 0, mx/2, colors="r", linestyles="dashed")
    
    ax[:set_xlim](x[1], x[end])
    ax[:set_ylim](0, mx)
#    tks = int(linspace(0, mx, 5))
#    ax[:yaxis][:set_ticks](tks)
    ax[:set_ylabel](L"$\sum_{m,n} |a_{m,n}|^2$ [a.u.]")
    ax[:set_xlabel](L"$\omega_0 [J]$")
    fig[:savefig]("fig_berry_int", bbox_inches="tight")
end


function plotreal(ω0::Float64; N=pm["N"], lim=div(pm["N"]-1,2), seed = 1234, random = 1)
    sz = div(N-1,2) - lim
    st = 1+sz
    en = N-sz
    P = setpump(; seed = seed, random = random)
    data = abs2(getass(genspmat(ω0), P))[st:en,st:en]./(N^2)
    fig, ax = plt.subplots(figsize=(4, 4))

    img = ax[:imshow](data, origin="upper", ColorMap("hot"), interpolation="none",
                extent=[-lim, lim, -lim, lim])

    ax[:set_ylim](-lim, lim)
    ax[:set_xlim](-lim, lim)
    ax[:set_xlabel](L"$m$")
    ax[:set_ylabel](L"$n$")

    tks = int(linspace(-lim,lim,5)) 
    ax[:xaxis][:set_ticks](tks)
    ax[:yaxis][:set_ticks](tks)

    cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
    cbar[:ax][:tick_params](labelsize=7)
    cbar[:set_label](L"$|a_{m,n}|^2$")

    fig[:savefig]("fig_berry_real", bbox_inches="tight")
end

function plotreal(ω0::Float64, n, m; N=pm["N"], lim=div(pm["N"]-1,2))
    sz = div(N-1,2) - lim
    st = 1+sz
    en = N-sz
    P = setpump(n,m)
    data = abs2(getass(genspmat(ω0), P))[st:en,st:en]./(N^2)
    fig, ax = plt.subplots(figsize=(4, 4))

    img = ax[:imshow](data, origin="upper", ColorMap("hot"), interpolation="none",
                extent=[-lim, lim, -lim, lim])

    ax[:set_ylim](-lim, lim)
    ax[:set_xlim](-lim, lim)
    ax[:set_xlabel](L"$m$")
    ax[:set_ylabel](L"$n$")

    tks = int(linspace(-lim,lim,5)) 
    ax[:xaxis][:set_ticks](tks)
    ax[:yaxis][:set_ticks](tks)

    cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
    cbar[:ax][:tick_params](labelsize=7)
    cbar[:set_label](L"$|a_{m,n}|^2$")

    fig[:savefig]("fig_berry_real", bbox_inches="tight")
end


function plotbz(ω0::Float64; N=pm["N"], seed = 1234, random = 1)
    P = setpump(; seed = seed, random = random)
    data = getass(genspmat(ω0), P)
    x = 2π*DSP.fftshift(DSP.fftfreq(N)) 
    databz = abs2(fftshift(fft(data./(N^2))))
    fig, ax = plt.subplots(figsize=(4, 4))

    img = ax[:imshow](databz, origin="upper", ColorMap("hot"), interpolation="none",
                                           extent=[x[1], x[end], x[1], x[end]])

    ax[:set_ylim](x[1], x[end])
    ax[:set_xlim](x[1], x[end])
    ax[:set_xlabel](L"$p_x$")
    ax[:set_ylabel](L"$p_y$")

    tks = [-3., -1.5, 0, 1.5, 3.]
    ax[:xaxis][:set_ticks](tks)
    ax[:yaxis][:set_ticks](tks)

    cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
    cbar[:ax][:tick_params](labelsize=7)

    fig[:savefig]("fig_berry_bz", bbox_inches="tight")
end

function plotbz(ω0::Float64,n,m; N=pm["N"])
    P = setpump(n,m)
    data = getass(genspmat(ω0), P)
    x = 2π*DSP.fftshift(DSP.fftfreq(N)) 
    databz = abs2(fftshift(fft(data./(N^2))))
    fig, ax = plt.subplots(figsize=(4, 4))

    img = ax[:imshow](databz, origin="upper", ColorMap("hot"), interpolation="none",
                                           extent=[x[1], x[end], x[1], x[end]])

    ax[:set_ylim](x[1], x[end])
    ax[:set_xlim](x[1], x[end])
    ax[:set_xlabel](L"$p_x$")
    ax[:set_ylabel](L"$p_y$")

    tks = [-3., -1.5, 0, 1.5, 3.]
    ax[:xaxis][:set_ticks](tks)
    ax[:yaxis][:set_ticks](tks)

    cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
    cbar[:ax][:tick_params](labelsize=7)

    fig[:savefig]("fig_berry_bz", bbox_inches="tight")
end


function foo(n,l,N)
    for i in 1:l
        idx = i
        while idx <= N
            print(idx," ")
            idx += l
        end
        println()
    end
end


function plotmbz(ω0::Float64; N=pm["N"], seed = 1234, random = 1)
    P = setpump(; seed = seed, random = random)
    data = getass(genspmat(ω0), P)
    x = 2π*DSP.fftshift(DSP.fftfreq(N)) 
    databz = abs2(fftshift(fft(data./(N^2))))

    n = int(1/pm["α"])
    l = div(N-1, n)

    V = Array(Float64,N,l)
    

    for i in 1:l
        idx = i
        while idx <= N
            V[:,i] += databz[:,idx]
            idx += l
        end
    end

    datambz = V

    fig, ax = plt.subplots(figsize=(4, 4))

    img = ax[:imshow](datambz, origin="upper", ColorMap("hot"), interpolation="none",
                                           extent=[-π*pm["α"], π*pm["α"], x[1], x[end]], aspect=pm["α"])

    ax[:set_ylim](x[1], x[end])
    ax[:set_xlim](-π*pm["α"], π*pm["α"])
    ax[:set_xlabel](L"$p_x$")
    ax[:set_ylabel](L"$p_y$")

    tks = [-3., -1.5, 0, 1.5, 3.]
    ax[:yaxis][:set_ticks](tks)

    cbar = fig[:colorbar](img, shrink=0.8, aspect=20, fraction=.12,pad=.02)
    cbar[:ax][:tick_params](labelsize=7)

    fig[:savefig]("fig_berry_mbz", bbox_inches="tight")
end

function genham(; N=pm["N"], α=pm["α"], κ=pm["κ"])
    iseven(N) && error("N must be odd")

    # Determine memory usage
    nz = countnonzeros(; N=N)

    # Preallocate
    I = Array(Int64,nz)
    J = Array(Int64,nz)
    V = Array(Complex{Float64},nz)

    function setnzelem(i,n,m; pos="self")
        if pos=="left"
            k += 1
            J[k] = i-N; I[k] = i; V[k] = -1
        elseif pos=="right"
            k += 1
            J[k] = i+N; I[k] = i; V[k] = -1
        elseif pos=="up"
            k += 1
            J[k] = i-1; I[k] = i; V[k] = -exp(-im*2π*α*m)
        elseif pos=="down"
            k += 1
            J[k] = i+1; I[k] = i; V[k] = -exp(im*2π*α*m)
        elseif pos=="self"
            k += 1
            J[k] = i; I[k] = i; V[k] = 1/2*κ*(m^2+n^2)
        end
    end
            
    # maximum value of m or n indices
    maxm = div(N-1,2)

    k = 0
    for i in 1:N^2
        m = getm(i; N=N)
        n = getn(i; N=N)
        #self interaction is always present
        setnzelem(i,n,m)
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



function alleigs(; N=pm["N"], α=pm["α"], κ=pm["κ"])
    M = genham(; N=N, α=α, κ=κ)
    ans = eigs(M; nev=N^2, which=:SR, ritzvec=false)
    real(ans[1])
end

const theeigs = alleigs()

end

#using Autoreload
#arequire("BerryPhase.jl")
#areload()

#BerryPhase.plotintensity()
#BerryPhase.plotintensity(0,0)
#BerryPhase.plotreal(BerryPhase.cω0)
#BerryPhase.plotbz(BerryPhase.cω0)
#BerryPhase.plotmbz(BerryPhase.cω0)
#BerryPhase.pm



#M-x set-input-method RET TeX 
#View available symbols by executing M-x describe-input-method RET TeX 











