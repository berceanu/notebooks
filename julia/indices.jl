#m(i,N)=div(i-1,N)+1
#n(i,N)=rem(i-1,N)+1
#i(m,n,N)=(m-1)*N+(n-1)+1

m(i,N)=div(i-1,N)-div(N-1,2)
n(i,N)=div(N-1,2)-rem(i-1,N)
indices(i,N) = (m(i,N), n(i,N))
i(m,n,N)=(m+div(N-1,2))*N+(div(N-1,2)-n)+1


N=3
for k in -div(N-1,2):div(N-1,2) #m
    for l in div(N-1,2):-1:-div(N-1,2) #n
        print(i(k,l,N)," ")
    end
end


for j in 1:N^2
    print(indices(j,N), " ")
end


# $$a_{m+1,n}+a_{m-1,n}+m(a_{m,n+1}+a_{m,n-1})+(m^{2}+n^{2})a_{m,n}=f_{m,n}$$
for k in -div(N-1,2):div(N-1,2) #m
    for l in div(N-1,2):-1:-div(N-1,2) #n
        println("a($(k+1),$l)+a($(k-1),$l)+$k(a($k,$(l+1))+a($k,$(l-1)))+$(l^2+k^2)a($k,$l)=f($k,$l)") #l-m, k-n
    end
end


# $$a_{i+N}+a_{i-N}+m(i,N)(a_{i-1}+a_{i+1})+(m^2(i,N)+n^2(i,N))a_i = f_i$$
for j in 1:N^2
    println("a($(j+N))+a($(j-N))+$(m(j,N))(a($(j-1))+a($(j+1)))+$(m(j,N)^2+n(j,N)^2)a($j)=f($j)")
end


#sorted
println()
for j in 1:N^2
    println("a($(j-N))+$(m(j,N))a($(j-1))+$(m(j,N)^2+n(j,N)^2)a($j)+$(m(j,N))a($(j+1))+a($(j+N))=f($j)")
end


# sorted + step functions
#θmin(idx) = idx > 0 ? 1 : 0
#θmax(idx,max) = idx > max ? 0 : 1
#for j in 1:N^2
#    println("$(θmin(j-N))a($(j-N))+$(θmin(j-1))$(m(j,N))a($(j-1))+$(m(j,N)^2+n(j,N)^2)a($j)+$(θmax(j+1,N^2))$(m(j,N))a($(j+1))+$(θmax(j+N,N^2))a($(j+N))=f($j)")
#end

getm(i,N)=div(i-1,N)-div(N-1,2)
getn(i,N)=div(N-1,2)-rem(i-1,N)
function genmat(N)
    if iseven(N)
        error("even N not allowed")
    end
    mat = zeros(Int64, N^2, N^2)
    for j in 1:N^2
        m = getm(j,N)
        n = getn(j,N)
        if j > N
        mat[j,j-N] = 1
        mat[j,j-1] = m
        elseif j > 1
            mat[j,j-1] = m
        end
    mat[j,j] = m*m+n*n
    if j <= (N-1)*N #N^2-N
            mat[j,j+1]=m
        mat[j,j+N]=1
        elseif j <= (N-1)*(N+1) #N^2-1
        mat[j,j+1]=m
    end
    end
    return mat
end

#$$a_{i-N}+e^{-i2\pi\alpha m}a_{i-1}+\left[\omega_{0}+i\gamma-\frac{1}{2}\kappa(m^{2}+n^{2})\right]a_{i}+e^{i2\pi\alpha m}a_{i+1}+a_{i+N}=fe^{i\phi_{i}}$$

function genspmat(N,α,ω0,γ,κ)
    if iseven(N)
        error("even N not allowed")
    end
    I = Int64[]
    J = Int64[]
    V = Complex{Float64}[]
    for i in 1:N^2
        m = getm(i,N)
        n = getn(i,N)
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


function setpump(f, N)
    # generate matrix of random phases in interval [0,2π)
    ϕ = 2π .* rand(N, N)
    f .* exp(im .* ϕ)
end


function getas(S, fmat)
    N = size(fmat)[1]
    fvec = reshape(fmat, N^2)
    reshape(S\fvec, N, N)
end

function genspmat2(N)
    iseven(N) && error("even N not allowed")
    
    # Determine memory usage
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

    # Preallocate
    I = Array(Int64,k)
    J = Array(Int64,k)
    V = Array(Int64,k)

    k = 0
    for i in 1:N^2
        m = getm(i,N)
        n = getn(i,N)
        k += 1
        J[k] = i; I[k] = i; V[k] = m*m+n*n
        if i > N
            k += 1
            J[k] = i-N; I[k] = i; V[k] = 1
            k += 1
            J[k] = i-1; I[k] = i; V[k] = m
        elseif i > 1
            k += 1
            J[k] = i-1; I[k] = i; V[k] = m
        end
        if i <= (N-1)*N
            k += 1
            J[k] = i+1; I[k] = i; V[k] = m
            k += 1
            J[k] = i+N; I[k] = i; V[k] = 1
        elseif i <= (N-1)*(N+1)
            k += 1
            J[k] = i+1; I[k] = i; V[k] = m
        end
    end
    return sparse(I,J,V)
end

using Base.Test
testmat = [ 2 -1  0  1 0 0 0 0 0;
           -1  1 -1  0 1 0 0 0 0;
        0 -1  2 -1 0 1 0 0 0;
        1  0  0  1 0 0 1 0 0;
        0  1  0  0 0 0 0 1 0;
        0  0  1  0 0 1 0 0 1;
        0  0  0  1 0 1 2 1 0;
        0  0  0  0 1 0 1 1 1;
        0  0  0  0 0 1 0 1 2]

@test genmat(3) == testmat
@test full(genspmat(3)) == testmat
@test genmat(101) == full(genspmat(101))

@time genmat(101);
#elapsed time: 0.212919415 seconds (832483344 bytes allocated)
@time genspmat(101);
#elapsed time: 0.005341752 seconds (8746272 bytes allocated)

