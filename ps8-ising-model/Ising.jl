module Ising

export IsingModel, update!, simulate!

"""
Ising model with periodic boundary conditions and arbitrary number of dimensions.

To simplify the calculations, the units are set in way that J=1 and kB=1
(the boltzman constant). So, the energy is in units of J and the temperature
is in units of J/kB. β can be thought of as J/(kB*T) or 1/T in this unit system.

`dims` in the constructor takes the number of particles in each dimension,
so that the number of elements of `dims` is equal to the dimensionality of the system.

The simulation here utilizes the Metropolis Monte Carlo method. Also, a checkerboard
coloring pattern is used for flipping many spins at the same time
"""
mutable struct IsingModel{N}
    spins::Array{Int8, N}
    energy::Float64
    magnetization::Float64

    β::Float64
    mass::Float64
    dims::NTuple{N, Integer}

    # pre-calculated values for possible state changes
    _probs::Vector{Float64}

    # spins broken down into a checkerboard pattern. spins of each color
    # do not interact with each other, so they can be flipped simultatiously
    _black::SubArray{Int8, 1, Vector{Int8}, Tuple{Vector{Int64}}, false} # even index sum
    _white::SubArray{Int8, 1, Vector{Int8}, Tuple{Vector{Int64}}, false} # odd index sum

    _blackneighbors::SubArray{Int8, 2, Vector{Int8}, Tuple{Matrix{Int64}}, false}
    _whiteneighbors::SubArray{Int8, 2, Vector{Int8}, Tuple{Matrix{Int64}}, false}

    function IsingModel(β, dims::Integer...)
        spins = rand([Int8(-1), Int8(1)], dims...)

        mass = prod(dims)
        indices = CartesianIndices(spins)
        linindices = LinearIndices(spins)
        neighbors = _get_neighbors(indices, linindices, dims, mass)

        energy = -sum(reshape(spins, mass, 1) .* sum(spins[neighbors], dims=2)) / 2
        magnetization = sum(spins) / mass
        probs = exp.(-4β .* collect(-length(dims):length(dims)))

        isblack = sum.(Tuple.(indices)) .% 2 .== 0
        blackindices = linindices[findall(isblack)]
        whiteindices = linindices[findall(map(!, isblack))]

        black = @view spins[blackindices]
        white = @view spins[whiteindices]
        blackneighbors = @view spins[neighbors[blackindices, :]]
        whiteneighbors = @view spins[neighbors[whiteindices, :]]

        return new{length(dims)}(spins, energy, magnetization, β, mass, dims,
            probs, black, white, blackneighbors, whiteneighbors)
    end
end

"""
get the indices of the neighbors of every point in `indices`
"""
@inline function _get_neighbors(indices::CartesianIndices, linindices::LinearIndices,
        dims::Tuple{Vararg{Integer}}, mass::Integer)
    neighborindices = Matrix{Int}(undef, mass, 0)
    for dim in 1:length(dims)
        basevec = zeros(Integer, length(dims))
        basevec[dim] = 1
        step = CartesianIndex(basevec...)
        basevec[dim] = dims[dim]
        modulus = CartesianIndex(basevec...)

        # collecting the CartesianIndices into a vector to enable setindex
        nextneighbors = collect(indices .+ step)
        prevneighbors = collect(indices .- step)
        # referencing fixes broadcasting issues
        selectdim(prevneighbors, dim, 1) .+= Ref(modulus)
        selectdim(nextneighbors, dim, dims[dim]) .-= Ref(modulus)
        neighborindices = hcat(
            neighborindices,
            reshape(linindices[prevneighbors], mass, 1), 
            reshape(linindices[nextneighbors], mass, 1))
    end
    return neighborindices
end

"""
Update an `IsingModel` for one time step, where all spins are updated once.
The operations are vectorized on the two "black" and "white" indices.
"""
function update!(ising::IsingModel)
    blackΔEs = vec(2 * ising._black .* sum(ising._blackneighbors, dims=2))
    blackprobs = ising._probs[blackΔEs .÷ 4 .+ (length(ising.dims) + 1)]
    flipblack = rand(length(blackprobs)) .<= blackprobs
    ising._black[flipblack] *= -1

    whiteΔEs = vec(2 * ising._white .* sum(ising._whiteneighbors, dims=2))
    whiteprobs = ising._probs[whiteΔEs .÷ 4 .+ (length(ising.dims) + 1)]
    flipwhite = rand(length(whiteprobs)) .<= whiteprobs
    ising._white[flipwhite] *= -1

    ising.magnetization += 2 / ising.mass * (
        sum(ising._black[flipblack]) + sum(ising._white[flipwhite]))
    ising.energy += sum(blackΔEs[flipblack]) + sum(whiteΔEs[flipwhite])
end

"""
Update an `IsingModel` for `time` steps. If `fullstats` is `false`, calculate and return
the mean and mean squared energy and magnetization (in the order ⟨E⟩, ⟨m⟩, ⟨E²⟩, ⟨m²⟩)
over the simulation time period. If `fullstats` is `true`, record and return the energy
and magnetization for every time step of the simulation (in the order Es, ms)

note: m is averaged over the absolute value
"""
function simulate!(isingmodel::IsingModel, time::Integer; fullstats::Bool=false)
    if fullstats
        energy = Vector{Float64}(undef, time)
        magnetization = Vector{Float64}(undef, time)

        for t in 1:time
            update!(isingmodel)
            energy[t] = isingmodel.energy
            magnetization[t] = isingmodel.magnetization
        end

        return energy, magnetization
    else
        Ē, m̄, Ē², m̄² = zeros(4)

        for _ in 1:time
            update!(isingmodel)
            Ē += isingmodel.energy / time
            m̄ += abs.(isingmodel.magnetization) / time
            Ē² += isingmodel.energy^2 / time
            m̄² += isingmodel.magnetization^2 / time
        end

        return Ē, m̄, Ē², m̄²
    end
end

using Statistics: mean, var

"""
    Calculate the correlation function of an `IsingModel` along the dimension `dim`,
    taking into account the periodic boundary conditions.
"""
function correlation(ising::IsingModel; dim=1)
    variance = var(ising.spins, mean=ising.magnetization)
    if variance == 0
        return zeros(ceil(Integer, size(ising.spins, dim) / 2))
    end

    correlations = Vector{Float64}(undef, ceil(Integer, size(ising.spins, dim) / 2))
    shiftvec = zeros(length(size(ising.spins)))
    for shift in 1:length(correlations)
        shiftvec[dim] = shift
        correlations[shift] = mean(ising.spins .* circshift(ising.spins, shiftvec))
    end

    # demean and normalize
    correlations .-= ising.magnetization ^ 2
    correlations ./= variance

    return correlations
end

"""
    Calculate the correlation length of an `IsingModel`, taking into account the periodic
    boundary conditions. Correlation data smaller than `thresh` is discarded (this is
    considered as "noise"). To counteract cases where the correlation length in one
    direction diverges, in these cases the correlation length is calculated for two
    directions and the minimum of the two correlation lengths is returned.
"""
function corlen(ising::IsingModel; thresh::Float64 = 0.1)
    correlations = correlation(ising, dim=1)
    threshindex = findfirst(x -> abs(x) <= thresh, correlations)

    if isnothing(threshindex)
        threshindex = length(correlations)

        # newcorr = correlation(ising, dim=2)
        # newthresh = findfirst(x -> abs(x) <= thresh, newcorr)
        # if isnothing(newthresh)
        #     newthresh = length(newcorr)
        # elseif newthresh < 3
        #     return 0.0
        # end

        # negativeξ⁻¹ = ([collect(1:threshindex) ones(threshindex)]
        #     \ log.(abs.(correlations[1:threshindex])))[1]
        # ξ = isnan(negativeξ⁻¹) ? 0.0 : 1/negativeξ⁻¹
        # if ξ < 0.0
        #     return 0.0
        # end

        # newnegativeξ⁻¹ = ([collect(1:newthresh) ones(newthresh)]
        #     \ log.(abs.(newcorr[1:newthresh])))[1]
        # newξ = isnan(newnegativeξ⁻¹) ? 0.0 : 1/newnegativeξ⁻¹
        # if newξ < 0.0
        #     return 0.0
        # end

        # return min(ξ, newξ)
    elseif threshindex < 3
        return 0.0
    end

    negativeξ⁻¹ = ([collect(1:threshindex) ones(threshindex)]
        \ log.(abs.(correlations[1:threshindex])))[1]
    if isnan(negativeξ⁻¹)
        return 0.0
    end

    ξ = -1/negativeξ⁻¹
    if ξ < 0
        return 0.0
    end

    return ξ
end

end
