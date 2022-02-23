using Distances

# compute dft spectra at subset of freqs
# (in production could use fft)
ω = unit .* collect(1:40)
FT = [exp.(-2*π*im * k * n / N) for k = ω, n = 0 : N-1];

peptide_spectra = zeros(ComplexF64, length(ω), nsamples, npep);

for pn = 1 : npep
    peptide_spectra[:,:,pn] .= FT * peptides[pn] ./ N
end

# use TMA clustering:
spectra = reshape(peptide_spectra, size(FT,1), npep*nsamples);
SD = pairwise(Euclidean(), spectra, dims=2); # pairwise Euclidean distances of columns
# build δ-Rips graph
δ = .35
A = delta_rips(SD, δ)
#check sparsity
length(findall(A))/length(A[:])
A = sparse(A)

σ = .05;
K = vec(sum(exp.(- (SD./δ).^2 ), dims = 2))
K ./ sum(K)

S, PD = TMA(A,K);
# test clustering accuracy

# compute and store centroids of each cluster

# generate protien sequences

# compute sdft and assign to clusters based on centroids above

# now run TMA on sdft spectra and compare 