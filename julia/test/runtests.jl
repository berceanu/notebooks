using BerryPhase
using Base.Test

const N=3
const tol=1e-3

P = ones(N,N)
S = genspmat(1.; N=N, α=1., γ=1., κ=1.)
X = getass(S,P)

testX = 
[0.236209-0.404665im  0.294335-0.0011107im   0.23732-0.11033im ;
    0.301-0.235098im  0.140874+0.331544im      0.301-0.235098im;
  0.23732-0.11033im   0.294335-0.0011107im  0.236209-0.404665im];

@test norm(testX - X) < tol

println("All looks OK")
