# Julia implementation of ToMaTo algorithm of Frédéric Chazal, Leonidas J. Guibas, Steve Y. Oudot, and Primoz Skraba. 
# 2013. Persistence-Based Clustering in Riemannian Manifolds. J. ACM 60, 6, 
# Article 41 (November 2013), 38 pages. DOI:https://doi.org/10.1145/2535927

# © 2021, Northwest Mathematics LLC, <consulting@northwestmath.com>

# Input is a simple graph and a density estimate on the vertices.  δ-Rips
# graph is reccomended.  For the density estimate, a Gaussian can be used,
# the authors reccommend a DTM estimate.

using SparseArrays
using LinearAlgebra
using DataStructures

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
    U = IntDisjointSets(n)
    g = zeros(Int,n)
    nclust = 0

    # intialize persistance diagram and tree
    PD = spzeros(n,2)
    tree = zeros(n,2)

    for i in I
        N = A.rowval[A.colptr[i]:(A.colptr[i+1]-1)]
        N = N[f[N].> f[i]]
        if isempty(N)
            # i is a peak in G
            PD[i,:] = [f[i],-10*f[I[end]]]
            nclust += 1
        else
            ~ , nk = findmax(f[N])
            g[i] = N[nk] #g[i] is approx gradient, neighbor with max f
            rᵢ = U.parents[g[i]]
            if J[rᵢ] >= J[i]
                error("J[rᵢ] >= J[i]")
            end
            # merge i into rᵢ component
            union!(U,rᵢ,i)
            if PD[i,1] !==0.0 # if a prior root with a birth time, record death time
                PD[i,2] = f[i]
                #if PD[i,1]<PD[i,2]
                #    error("birth time less than death time")
                #end
            end

            # now we compare approx gradients at all other neighbors to find
            # local max and merge to that root
            for j in N
                rⱼ = U.parents[j]
                if rⱼ !== rᵢ #&& minimum([f[rⱼ], f[rᵢ]]) < τ
                    if f[rᵢ] < f[rⱼ]
                        union!(U,rⱼ,rᵢ)
                        nclust -= 1
                        if PD[rᵢ,1] !==0.0
                            PD[rᵢ,2] = f[rᵢ]
                            #if PD[rᵢ,1]<PD[rᵢ,2]
                            #    error("birth time less than death time")
                            #end
                        end
                    else
                        union!(U,rᵢ,rⱼ)
                        nclust -= 1
                        if PD[rⱼ,1] !==0.0
                            PD[rⱼ,2] = f[rⱼ]
                            #if PD[rⱼ,1]<PD[rⱼ,2]
                            #    error("birth time less than death time")
                            #end
                        end
                    end
                    rᵢ = find_root!(U,rⱼ) #rᵢ set to new highest root
                end
            end
        end
        # record number of clusters at current level
        tree[J[i],:] .= [f[i], nclust]
    end
    # output is sets in U with root such that f[r] ≧ τ
    S = Vector{Int64}[]
    Noise = Int64[]
    # update roots of all points
    # (can this be done faster, or within UnionFind?) 
    # (This should be automatic I think, full linear cost even one time like below is not great...)
    for i = 1 : n
        find_root!(U,i)
    end
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

