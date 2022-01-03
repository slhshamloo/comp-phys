cd(dirname(@__FILE__))
include("MolDyn.jl")
using Statistics, DataFrames, CSV
datapath = "data/"

num = 100
sidelen = 30.0
datacount = 41

inittemps = collect(range(1.0, 3.0, length=datacount))
meanT, meanP, σT, σP = [Vector{Float64}(undef, datacount) for _ in 1:4]

for i in 1:datacount
    sim = MolDyn.MolSim(num, inittemps[i], sidelen, sidelen)
    MolDyn.alignleft!(sim)
    MolDyn.simulate!(sim, 4000, 0.01)
    # evolve the system to equilibrium
    # collect data
    _, temperature, pressure = MolDyn.simulate!(sim, 40000, 0.001)
    meanT[i] = mean(temperature)
    meanP[i] = mean(pressure)
    σT[i] = std(temperature)
    σP[i] = std(pressure)
end

# save data to file
df = DataFrame(
    "temperature mean" => meanT, "temperature std" => σT,
    "pressure mean" => meanP, "pressure std" => σP)
CSV.write(datapath * "vanderwaals-N$num-L$sidelen.csv", df)
