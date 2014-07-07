module Various

import Contour: Curve2

export getidx, fftfreq, vec2range, cropmat, sysbox

# parametrization of parabolic wavefronts for fitting
type parabola
    xpoints::Vector{Float64}
    ypoints::Vector{Float64}
end

paraby(M::Int64, t::Float64, offset::Float64) = 2pi*M*t + offset
parabx(M::Int64, t::Float64, offset::Float64, k0::Float64) = -pi*M/k0 + pi*M*t^2 + offset

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

# TODO: replace with upstream version
# port of numpy.fft.fftfreq
function fftfreq(n::Int64, d::Float64)
  N = fld(n-1,2)
  p1 = [0:N]
  p2 = [-fld(n,2):-1]
  return [p1, p2]/(d*n)
end

# convert a Vector to a FloatRange
function vec2range(v::Vector{Float64})
  issorted(v) || error("Not sorted")
  a = (v[end] - v[1])/(length(v)-1)
  for i in 2:length(v)
    isapprox(a, v[i]-v[i-1]) || error("Differences are not constant")
  end
  colon(v[1], a, v[end])
end

# crop a matrix in r or k space
function cropmat(m::Matrix, x::FloatRange{Float64}, y::FloatRange{Float64}; xcenter=0., ycenter=0., side=10.)
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


end
