# Julia implementation of ToMaTo algorithm of Frédéric Chazal, Leonidas J. Guibas, Steve Y. Oudot, and Primoz Skraba. 
# 2013. Persistence-Based Clustering in Riemannian Manifolds. J. ACM 60, 6, 
# Article 41 (November 2013), 38 pages. DOI:https://doi.org/10.1145/2535927

# © 2021, Northwest Mathematics LLC, <consulting@northwestmath.com>

# Input is a simple graph and a density estimate on the vertices.  δ-Rips
# graph is reccomended.  For the density estimate, a Gaussian can be used,
# the authors reccommend a DTM estimate.

using SparseArrays
using LinearAlgebra

# using my own implementation of union-find structure, can make faster at some point
# Julia package DataStructures.jl implementation is broken...
mutable struct UnionFind
    parents::Vector{Int}
    ranks::Vector{Int} 
    ngroups::Int
    UnionFind(n) = new(collect(1:n),ones(Int,n),n)
end

function TMA(A::SparseMatrixCSC, f::Vector{Float64}, τ::Float64 = Inf)

    # sort indices according to descending f value
    I = sortperm(f, rev = true)

    # inverse (is there a quicker way to do this?)
    J = zeros(Int,length(I))
    for j=1:length(I)
        J[I[j]]=j
    end

    # initialize clustering
    n = length(f)
    U = UnionFind(n)
    g = zeros(Int,n)

    # intialize persistance diagram and tree
    PD = spzeros(n,2)
    tree = zeros(n,2)

    for i in I
        N = A.rowval[A.colptr[i]:(A.colptr[i+1]-1)]
        N = N[f[N].> f[i]]
        if isempty(N)
            # i is a peak in G
            PD[i,:] = [f[i],-10*f[I[end]]]
        else
            ~ , nk = findmax(f[N])
            g[i] = N[nk] #g[i] is approx gradient, neighbor with max f
            rᵢ = U.parents[g[i]]
            if J[rᵢ] >= J[i]
                error("J[rᵢ] >= J[i]")
            end
            # merge i into rᵢ component
            U.parents[i] = rᵢ
            U.ranks[rᵢ] += 1
            U.ngroups -= 1
            if PD[i,1] !==0.0 # if a prior root with a birth time, record death time
                PD[i,2] = f[i]
                if PD[i,1]<PD[i,2]
                    error("birth time less than death time")
                end
            end

            # now we compare approx gradients at all other neighbors to find
            # local max and merge to that root
            for j in N
                rⱼ = U.parents[j]
                if rⱼ !== rᵢ && minimum([f[rⱼ], f[rᵢ]]) < τ
                    if f[rᵢ] < f[rⱼ]
                        #nroots = findall(U.parents .== rᵢ)
                        U.parents[U.parents .== rᵢ] .= rⱼ
                        #U.ranks[rⱼ] += length(nroots)
                        U.ngroups -= 1 
                        if PD[rᵢ,1] !==0.0
                            PD[rᵢ,2] = f[rᵢ]
                            if PD[rᵢ,1]<PD[rᵢ,2]
                                error("birth time less than death time")
                            end
                        end
                    else
                        #nroots = findall(U.parents .== rⱼ)
                        U.parents[U.parents .== rⱼ] .= rᵢ
                        #U.ranks[rᵢ] += length(nroots)
                        U.ngroups -= 1 
                        if PD[rⱼ,1] !==0.0
                            PD[rⱼ,2] = f[rⱼ]
                            if PD[rⱼ,1]<PD[rⱼ,2]
                                error("birth time less than death time")
                            end
                        end
                    end
                    rᵢ = U.parents[rⱼ] #rᵢ set to new highest root
                end
            end
        end
        # record number of clusters at current level
        roots = unique(U.parents)
        num_clust = length(findall(f[roots] .>= f[i]))
        tree[J[i],:] .= [f[i], num_clust]
    end
    # output is sets in U with root such that f[r] ≧ τ
    S = Vector{Int64}[]
    Noise = Int64[]
    roots = unique(U.parents)
    for r in roots
        if τ < Inf
            if f[r] >= τ
                push!(S,findall(U.parents .== r))
            else
                union!(Noise,findall(U.parents .== r))
            end
        else
            push!(S,findall(U.parents .== r))
        end
    end
    push!(S,Noise)
    return S, PD, tree
end



# δ-Rips graph 
# input is array of Distances
using LinearAlgebra
function delta_rips(D::Matrix{Float64}, δ::Float64)

    A = ones(size(D))
    A[D.>δ] .= 0
    A[diagind(A)] .= 0
    A = BitArray(A)

    return(A)
end

