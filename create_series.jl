
using Random
include("./TMA.jl")
include("./sdft.jl")
# First we generate synthetic peptide series that qualitatively match those in the bioarxiv paper

# define parameters for building our samples
mutable struct SeriesParams
    length :: Int64             # length of series samples of 
    peptides:: Int64            # number of peptides (distinct sample types)
    avg_times :: Vector{Float64}# parameters for exponential rv's defining floresce times 
    avg_dur :: Vector{Float64}  # parameters for exponential rv's defning duration of floresce events
    avg_ampl :: Vector{Float64} # μ for amplitudes ∼ N(μ,1.0)
    noise :: Tuple{Float64,Float64}# μ and σ for background noise
end

# generate random peptide series
function peptide_series(sparams::SeriesParams, NS::Int64)
    N = sparams.length
    npep = sparams.peptides;
    avg_times = sparams.avg_times;
    avg_dur = sparams.avg_dur;
    avg_ampl = sparams.avg_ampl;
    noise = sparams.noise;
    # output as array of npep arrays, each peptide sample array N x NS
    peptide = Vector{Matrix{Float64}}(undef, npep)
    
    # we can parallelize construction in a few ways, here we just loop over npeps
    for pn = 1 : npep
        Sample = zeros(N,NS)
        for j = 1 : NS
            # generate sequence of floresce events:
            # start time, duratin and amplitude
            # until we reach end of sample
            λ = avg_times[pn]
            α = avg_dur[pn]
            H = avg_ampl[pn]
            time = 1
            A = zeros(N)
            t = time + round(Int,-log(rand(1)[1])/λ) # exponential rv with mean λ
            # make sure first time ≥ 1
            a = round(Int, max(1.0, α + (α/2) *randn(1)[1])) # Gaussian rv with mean α and variance α/2
            time = t + a;
            while time < N
                amp = H + randn(1)[1]
                A[t:t + a] .= H;
                t = time + round(Int,-log(rand(1)[1])/λ) # exponential rv with mean λ
                a = round(Int, max(1.0, α + (α/2) *randn(1)[1])) # Gaussian rv with mean α and variance α/2
                time = t + a;
            end

            # generate background noise
            W = noise[1] .+ noise[2].*randn(N)

            # add floresce events to background noise
            Sample[:,j] = A + W;
        end
        peptide[pn] = Sample
    end
    return peptide
end

# define protein series structure
mutable struct protein
    series::Vector{Float64} #the measured series
    intervals::Vector{Tuple{Int64,Int64}} # intervals of florescence of each peptide in the sequence
    peptides::Vector{Int64} # indices of the peptides
end

# generate random protien series
function protein_series(sparams::SeriesParams, M::Int64, NS::Int64)
    npep = sparams.peptides;
    avg_times = sparams.avg_times;
    avg_dur = sparams.avg_dur;
    avg_ampl = sparams.avg_ampl;
    noise = sparams.noise;

    proteins = protein[]

    for pn = 1 : NS
        intervals = Tuple{Int64,Int64}[];
        peptides = Int64[];

        # generate background noise series
        sample = noise[1] .+ noise[2].*randn(M)

        # random number of peptides (minimum of 3)
        numpeps = rand(3:length(avg_times))

        # set intervals and durations to approximately fill series
        wait_time = M / (2 * numpeps)
        duration = M / (2 * numpeps)

        # select those at random from series sparams
        peps = rand(1:length(avg_times),numpeps)

        # generate random start times and durations,
        # generate sequences and insert iteratively
        Time = 1
        T = Time + round(Int,-log(rand(1)[1])/wait_time) # exponential rv with mean wait_time
            # make sure first time ≥ 1
        N = round(Int,-log(rand(1)[1])/duration) # exponential rv with mean duration
        Time = T + N
        for pep in peps
            if Time >=M
                break
            end
            λ = avg_times[pep]
            α = avg_dur[pep]
            H = avg_ampl[pep]
            time = 1
            A = zeros(N)
            t = time + round(Int,-log(rand(1)[1])/λ) # exponential rv with mean λ
            # make sure first time ≥ 1
            a = round(Int, max(1.0, α + (α/2) *randn(1)[1])) # Gaussian rv with mean α and variance α/2
            while time < N
                amp = H + randn(1)[1]
                A[t:t + a] .= H;
                t = time + round(Int,-log(rand(1)[1])/λ) # exponential rv with mean λ
                a = round(Int, max(1.0, α + (α/2) *randn(1)[1])) # Gaussian rv with mean α and variance α/2
                time = t + a;
            end
            # add floresce events to background noise
            sample[T : T + N] .+= A;

            #store interval and peptide index
            push!(intervals,(T,T+N))
            push!(peptides,pep)

            time = T + N;
        end
        push!(proteins,protein(sample,intervals,peptides))
    end
    # output series, times/indices and labels/types
    return proteins
end

# generate training peptide series
unit = 10
nsamples = 1000;
N = 120 * unit; 
npep = 5;
avg_times = (unit.*[5.0, 7.5, 10.0, 15.0, 20.0]).^-1; #seconds
avg_dur = (unit.*[.1, 1.0, 1.0, .5, 5.0]); #seconds
avg_ampl = [11.8, 6.9, 11.5, 7.8, 8.5];
noise = (2.0,.25);
sparams = SeriesParams(N, npep, avg_times, avg_dur, avg_ampl, noise);

peptides = peptide_series(sparams, nsamples);