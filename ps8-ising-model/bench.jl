cd(dirname(@__FILE__))
include("Ising.jl")
using BenchmarkTools
isingmodel = Ising.IsingModel(1, 64, 64)
# @benchmark Ising.update!(isingmodel)
Ising.simulate!(isingmodel, 1000)
@benchmark Ising.corlen(isingmodel)
