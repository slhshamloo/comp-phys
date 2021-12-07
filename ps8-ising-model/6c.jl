cd(dirname(@__FILE__))
include("Ising.jl");
using DataFrames, CSV, Statistics
using StatsBase: autocor
datapath = "data/2d/"

βmin, βmax, βcount = 0.1, 0.8, 51
βs = collect(range(βmin, βmax, length=βcount))
sidelen = 256

relaxtime = 10^4
runtime = 10^4

corr = zeros(βcount)
for i in 1:βcount
    isingmodel = Ising.IsingModel(βs[i], sidelen, sidelen)
    # evolve the system to equilibrium
    Ising.simulate!(isingmodel, relaxtime)
    # take data
    for _ in 1:runtime
        Ising.update!(isingmodel)
        corr[i] += Ising.corlen(isingmodel) / runtime
    end
end

# save data to file
df = DataFrame("beta" => βs, "corr len" => corr)
CSV.write(datapath * "full$sidelen-corr.csv", df)

βzoom = collect(range(0.422 - 0.05, 0.422 + 0.05, length=βcount))
corr = zeros(βcount)
for i in 1:βcount
    isingmodel = Ising.IsingModel(βzoom[i], sidelen, sidelen)
    # evolve the system to equilibrium
    Ising.simulate!(isingmodel, relaxtime)
    # take data
    for _ in 1:runtime
        Ising.update!(isingmodel)
        corr[i] += Ising.corlen(isingmodel) / runtime
    end
end

# save data to file
df = DataFrame("beta" => βzoom, "corr len" => corr)
CSV.write(datapath * "zoom$sidelen-corr.csv", df)
