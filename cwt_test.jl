using Distances, NearestNeighbors, Statistics


# define scaling function (up to a constant)
ψ(x) = (1 - 2x^2) * exp(-x^2)

# choose scales
α =collect(1:2:51)

# build wavelet operators
# (there are obviously much faster ways to do this)
# (we just do below for ease of exposition)

# array comprehension with Julia, fast and pretty ☻
Ψ = [ψ((i-j)/s) / s for i = 1 : N, j = 1 : N, s in α]
# so each row of the matrix product Ψ * f is a convuliution with scaled ψ

# compute wavelet spectra of each series
pepWS = zeros(N,length(α), nsamples, npep)

for a = 1 : length(α), np = 1 : npep
    pepWS[:,a,:,np] .= Ψ[:,:,a] * (peptides[np] .- mean(peptides[np],dims=1) ); # substract mean
end 

# filter (compute coordinates) each series
pep_coords = zeros(3,nsamples,npep)
# for use in clustering below
A = sparse(diagm(1 => ones(N-1), -1 => ones(N-1))) # graph is just integer lattice of points on line
for pn = 1 : npep, sn = 1 : nsamples
    # choose max or average scale for each t
    M,I = findmax(pepWS[:,:,sn,pn], dims = 2);
    S = α[(x->x[2]).(I)]
    # filter t by energy (average)
    # should really use clustering here, like in simple_test
    ind = findall(M.< 1.2*sum(M)/length(M))
    M[ind] .= 0.0
    S[ind] .= 0

    # find local maxima via clustering using TMA
    f = vec(M) / sum(M)
    C, PD, tree = TMA(A,f, 1e-10);
    
    # find index of maximum on each cluster
    indm = zeros(Int64,length(C)-1)
    for cn = 1 : length(C) - 1
        ~, m = findmax(f[C[cn]])
        indm[cn] = C[cn][m]
    end

    # determine average amplitude, scale (weighted by M value), and wait time
    amp = sum(M[indm]) / length(indm)
    scale = sum(S[indm].*M[indm]) / length(indm)
    wait = (indm[1] .+ sum(indm[2:end] - indm[1:end-1])) / length(indm)

    pep_coords[:,sn,pn] .= [amp, scale, wait]
    println(sn)
    println(pn)
end

###
# cluster and check
###

wpoints = reshape(pep_coords, (3,:))
NP = size(wpoints,2)

# compute distances of points
D = pairwise(Euclidean(), wpoints, dims = 2) # pairwise Euclidean distances of columns

# use knn to choose scale (NearestNeighbors.jl)
kdtree = KDTree(wpoints)
k = 5
idxs, dists = knn(kdtree, wpoints, k, true)

# set δ to average of distance of kth nearest neighbor
kdists = (x->x[k]).(dists)
δ = sum(kdists) / NP

# build δ-Rips graph
A = delta_rips(D, δ)

#check sparsity
length(findall(A))/length(A[:])
A = sparse(A)

# build density. We filter by I + A so that isolated points have lowest density
K = sum(exp.(- (D./δ).^2 ) .* (LinearAlgebra.I + A), dims = 2)
K = vec(K) / NP

# cluster using TMA
Cw, PD, tree = TMA(A, K);

sizes = sort(length.(Cw), rev = true);
# test on protein series