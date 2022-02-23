# compute sliding discrete Fourier transform of a series
# inputs are a vector (series) of values, a window length L, and number of frequencies to calculate
# TODO speed up

function sdft(f::Vector{Float64}, L::Int64, nfreqs::Int64 = L)
    m = length(f)
    F = zeros(ComplexF64,nfreqs, m)

    # pad f with L zeros
    f = vcat(zeros(L), f)

    # precompute L roots of unity
    roots = exp.((2 * π * im) .* collect(0:nfreqs-1) ./ L )
    roots .= round.(roots, digits = 10)

    # initialize first value, assuming 'f[0]' = 0    
    F[:,1] .= ( f[L] ) .* roots

    #iterate    
    for t = 2 : length(f) - L
        F[:,t] .= ( F[:,t-1] .+ ( f[t - 1 + L] - f[t - 1] ) ) .* roots
    end

    return F
end