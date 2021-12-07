cd(dirname(@__FILE__))
include("Ising.jl");
using DataFrames, CSV, Statistics
using StatsBase: autocor
datapath = "data/2d/"

βmin, βmax, βcount = 0.1, 0.8, 51
βs = collect(range(βmin, βmax, length=βcount))
sidelen = 256

relaxtime = 10^4
runtime = 5 * 10^4

βzoom = collect(range(0.422 - 0.05, 0.422 + 0.05, length=βcount))
Ēs, m̄s, Ē²s, m̄²s = [Float64[] for i in 1:4]

for β in βzoom
    isingmodel = Ising.IsingModel(β, sidelen, sidelen)
    # evolve the system to equilibrium
    Ising.simulate!(isingmodel, relaxtime)
    # take data
    Ē, m̄, Ē², m̄² = Ising.simulate!(isingmodel, runtime)
    push!(Ēs, Ē)
    push!(Ē²s, Ē²)
    push!(m̄s, m̄)
    push!(m̄²s, m̄²)
end

# save data to file
df = DataFrame(
    "beta" => βzoom,
    "mean energy" => Ēs, "mean mag" => m̄s,
    "mean sq energy" => Ē²s, "mean sq mag" => m̄²s)
CSV.write(datapath * "zoom$sidelen.csv", df)
