# here we try a simple implementation of the ideas from cwt_test

using Distances, NearestNeighbors, Statistics, StatsBase

# first we use clustering to separate background noise from signals, based on intensity

signal = deepcopy(peptides);
for pn = 1 : npep, sn = 1 : nsamples
    # build histogram (density) of values
    h = fit(Histogram, peptides[pn][:,sn], nbins = 100)
    f = h.weights
    f /= sum(f)

    # build graph and cluster
    Ah = sparse(diagm(1 => ones(length(f)-1), -1 => ones(length(f)-1))) # graph is just integer lattice of points on line
    C, PD, tree = TMA(Ah,f)

    # find level to get at least two clusters
    # depending on bin density, we may get a jump from 1 to 3 or more clusters
    # this can be caught and refined, but for now this is fine.
    τ = tree[findlast(tree[:,2] .< 3),1]
    Cf, ~, ~ = TMA(Ah, f, τ)
    
    # compute means of each cluster
    m1 = mean(Cf[1]); m2 = mean(Cf[2])

    # now add "noise values" to nearest cluster
    if length(Cf) == 3
        for j in Cf[3]
            abs(j - m1) < abs(j - m2) ? push!(Cf[1],j) : push!(Cf[2],j)
        end
    end

    #determine signal cluster as higher
    # this is largely redundant, but it may happen the signal is denser than background over a given window
    m1<m2 ? SC = Cf[2] : SC = Cf[1] 

    # we have the indices of the bins of the cluster, 
    # we need to convert these to indices of the series
    min_signal =  minimum(collect(h.edges[1])[SC])
    ind_signal = findall(peptides[pn][:,sn] .>= min_signal)

    # now filter out noise by 
    denoised = zeros(N)
    denoised[ind_signal] = peptides[pn][ind_signal,sn]
    signal[pn][:,sn] = denoised
end

# now with our denoised series we can determine our coordinates:
pep_coords = zeros(3,nsamples,npep)
for pn = 1 : npep, sn = 1 : nsamples
    s = signal[pn][:,sn]
    # first amplitude, just average of nonzeros
    amp = mean(s[s.>0])

    # for duration and number of events, we count directly
    s = sign.(s)
    starts = Int[]; ends = Int[];
    if s[1] == 1.0 
        push!(starts,1)
    end
    for j = 2 : N
        if s[j] == 1.0 && s[j-1] == 0.0
            push!(starts,j)
        elseif s[j] == 0.0 && s[j-1] == 1.0
            push!(ends, j-1)
        end
    end
    if s[N] == 1.0
        push!(ends,N)
    end
    count = length(starts)
    duration = mean(ends - starts)
    pep_coords[:,sn,pn] = [amp, duration, count]
end


###
# cluster and check
###

points = reshape(pep_coords, (3,:))
NP = size(points,2)
npoints = points ./ [norm(points[1,:]), norm(points[2,:]),norm(points[3,:])]

# compute distances of points
D = pairwise(Euclidean(), npoints, dims = 2) # pairwise Euclidean distances of columns

# use knn to choose scale (NearestNeighbors.jl)
kdtree = KDTree(npoints)
k = 20
idxs, dists = knn(kdtree, npoints, k, true)

# set δ to average of distance of kth nearest neighbor
kdists = (x->x[k]).(dists)
δ = sum(kdists) / NP

# build δ-Rips graph (could also use knn graph)
A = delta_rips(D, δ)

#check sparsity
length(findall(A))/length(A[:])
A = sparse(A)

# build density. We filter by I + A so that isolated points have lowest density
K = sum(exp.(- (D./δ).^2 ) .* (LinearAlgebra.I + A), dims = 2)
K = vec(K) / NP

# cluster using TMA
C, PD, tree = TMA(A, K);

sizes = sort(length.(C), rev = true);
# test on protein series
