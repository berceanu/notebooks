module Various

import Contour: Curve2, ContourLevel
import DSP: fftshift, fftfreq
import PyCall: PyObject
import Images: imfilter_gaussian

export getidx, vec2range, cropmat, sysbox, gettrunccrop, sider, sidek, coordinates, torad, λtoε, εtoλ, readdata, plotcontours, filtergauss

# parametrization of parabolic wavefronts for fitting
type parabola
    xpoints::Vector{Float64}
    ypoints::Vector{Float64}
end

paraby(M::Int64, t::Float64, offset::Float64) = 2pi*M*t + offset
parabx(M::Int64, t::Float64, offset::Float64, k0::Float64) = -pi*M/k0 + pi*M*t^2 + offset

# gaussian filtering
function filtergauss(data::Matrix{Float64}, sigma:Vector{Float64})
    gaussdata = imfilter_gaussian(data, sigma)
    gaussdata, data - gaussdata
end

# plotting extracted contour
function plotcontours(c::ContourLevel, ax::PyObject)
    for line in c.lines # line is a Curve2, which is basically a wrapper around a Vector{Vector2}
        xs, ys = coordinates(line)
        ax[:plot](xs, ys, "y-")
    end
end

# TODO: replace with upstream version
# for extracting the vertices of contours
function coordinates(c::Curve2)
    N = length(c.vertices)
    xlist = Array(Float64,N)
    ylist = Array(Float64,N) 

    for (i,v) in enumerate(c.vertices)
        xlist[i] = v[1]
        ylist[i] = v[2]
    end
    xlist, ylist
end

type sysbox{T}
    y::FloatRange{T}
    x::FloatRange{T}
    ky::FloatRange{T}
    kx::FloatRange{T}
end

# get the index of a certain x (or k) value in an array
getidx(pos, len, Δ) = iround((len + pos)/Δ) + 1

# deprecated
#function fftfreq(n::Int64, d::Float64)
  #N = fld(n-1,2)
  #p1 = [0:N]
  #p2 = [-fld(n,2):-1]
  #return [p1, p2]/(d*n)
#end

# deprecated
# convert a Vector to a FloatRange
function vec2range(v::Vector{Float64})
  issorted(v) || error("Not sorted")
  a = (v[end] - v[1])/(length(v)-1)
  for i in 2:length(v)
    isapprox(a, v[i]-v[i-1]) || error("Differences are not constant")
  end
  colon(v[1], a, v[end])
end

function sider(;l=70., n=256)
	Δ = 2l/(n-1)
	-l:Δ:l
end

sidek(;l=70., n=256) = 2pi*fftshift(fftfreq(n, (n-1)/2l))

# reading data from full numerics
function readdata(path::String, file::String, script::String)
	filepath = path*file
	scriptpath = path*script
	run(`$scriptpath $filepath`)
	data = readdlm("$filepath.copy", Float64)
	run(`rm $filepath.copy`)
	return data
end

# reading and rescaling experimental data
function getexpfile(filename::ASCIIString)
    m = readdlm(filename, '\t', Float64, '\n')
    (mmin, mmax) = extrema(m)
    m = m .- mmin
    m[m .< 0.] = 0.
    m *= 255.0/(mmax - mmin)
    m[m .> 255.] = 255.
    return m    
end

# truncating experimental data
function truncdata(m::Matrix{Float64}; cmin=0., cmax=255.)
    m[m .> cmax] = cmax
    m[m .< cmin] = cmin
    return m
end

# crop a matrix in r or k space
function cropmat(m::Matrix, x::FloatRange{Float64}, y::FloatRange{Float64}; xcenter=0., ycenter=0., side=35.)
    lenx = maximum(x)
    stepx = step(x)
    leny = maximum(y)
    stepy = step(y)
    #coordinates of square xl, xr, yb, yt
    coords = Float64[xcenter - side, xcenter + side, ycenter - side, ycenter + side]
    #indices of square idxxl, idxxr, idxyb, idxyt
    idxs = map(getidx, coords, [lenx, lenx, leny, leny], [stepx, stepx, stepy, stepy])
    return (m[idxs[3]:idxs[4],idxs[1]:idxs[2]], x[idxs[1]:idxs[2]], y[idxs[3]:idxs[4]])
end

function gettrunccrop(filename::ASCIIString, x::FloatRange{Float64}, y::FloatRange{Float64}; cmax=180., xcenter=0., ycenter=0., side=35.)
    data = getexpfile(filename)
    tdata = truncdata(data; cmax=cmax)
    ctdata, cx, cy = cropmat(tdata, x, y; xcenter=xcenter, ycenter=ycenter, side=side)
end

# convert angle in degrees to radians
torad(angle::Float64) = 2pi*angle/360.

#converts nm to eV
λtoε(λ::Float64) = 1239.84193/λ

#converts eV to nm
εtoλ(ε::Float64) = 1239.84193/ε

end
