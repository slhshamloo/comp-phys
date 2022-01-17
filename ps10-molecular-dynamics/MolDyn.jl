module MolDyn

export MolSim, update!, simulate!, alignleft!
export get_temperature, get_potential, get_pressure

using Statistics: mean

"""
A molecular dynamics simulation with periodic boundary conditions in reduced units.
The Lennard-Jones potential is used with the velocity verlet integration method.

`pos` and `vel` hold the position and velocity vectors of each particle respectively;
each column represents one particle and each row holds one component of the vectors.

`dim` is the number of dimentions of the system.

the center of mass of the system is fixed (average velocities is equal to zero) to
avoid measuring extra temperature by calculating the average kinetic energy.

# Constructor Arguments
- `num`: the number of particles
- `inittemp`: Initial temperature of the system. Controls the initial velocities.
    (temperature is defined as the average kinetic energy of each degree of freedom in
    the center of mass frame of reference. the initial velocities are uniformly generated
    in a way that the average kinetic energy of the system is equal to `inittemp` after
    fixing the center of mass)
- `dims`: the size of the container in each direction.
"""
mutable struct MolSim{T<:AbstractFloat}
    num::Integer
    volume::T
    dim::Integer
    sidelens::Vector{T}
    pos::Matrix{T}
    vel::Matrix{T}

    function MolSim(num::Integer, inittemp::T, sidelens::T...) where {T<:AbstractFloat}
        volume = prod(sidelens)
        dim = length(sidelens)
        sidelens = collect(sidelens)

        # positions must be inside the container
        pos = rand(T, dim, num) .* sidelens

        # generate random velocities, fix the center of mass and scale the velocities
        # so that the temperature is equal to inittemp
        vel = rand(T, dim, num) .- 0.5
        vel .-= mean(vel, dims=2)
        vel *= √(inittemp / mean(x -> x^2, vel))

        return new{T}(num, volume, dim, sidelens, pos, vel)
    end
end

"""
Place the particles of the system in a simple cubic lattice in the "left" side of the
container, which is defined as the first half of the first dimention of the container.
"""
function alignleft!(sim::MolSim)
    xdim = sim.sidelens[1] / 2
    # ycount ^ (sim.dim - 1) * xcount ≈ sim.num
    xcount = ceil(Integer, (2 * sim.num) ^ (1/sim.dim) / 2)
    ycount = 2 * xcount

    xs = collect(range(0.05 * xdim, 0.95 * xdim, length=xcount))
    sim.pos[1, :] .= repeat(xs, outer = ycount ^ (sim.dim - 1))[1:sim.num]

    for i in 2:sim.dim
        ys = collect(range(0.01 * sim.sidelens[i], 0.99 * sim.sidelens[i], length=ycount))
        sim.pos[i, :] .= repeat(ys,
            inner = xcount * ycount ^ (i - 2),
            outer = ycount ^ (sim.dim - i))[1:sim.num]
    end
end

"""
Update the molecular dynamics simulation for one step and return the potential energy,
temperature and pressure after update.
"""
function update!(sim::MolSim{T}, stepsize::T) where {T<:AbstractFloat}
    prevforce, _, _ = _calculate_vanderwaals(sim)
    _, potential, virial = _velverlet!(sim, prevforce, stepsize)
    _apply_boundary_conditions!(sim)

    velsquaredsum = sum(x -> x^2, sim.vel)
    pressure = (velsquaredsum + virial) / (sim.dim * sim.volume)
    temperature = velsquaredsum / (sim.dim * sim.num)

    return potential, temperature, pressure
end

"""
Run the molecular dynamics simulation for `steps` number of steps, where `stepsize` is the
duration of each step, so that the total simulation time (in reduced units) is
`steps * stepsize`.

returns the potential energy, temperature, and pressure for each step.
"""
function simulate!(sim::MolSim{T}, steps::Integer, stepsize::T) where {T<:AbstractFloat}
    potentials, temperatures, pressures = [Vector{T}(undef, steps) for _ in 1:3]

    # calculate initial force
    prevforce, _, _ = _calculate_vanderwaals(sim)

    for n in 1:steps
        prevforce, potentials[n], virial = _velverlet!(sim, prevforce, stepsize)
        _apply_boundary_conditions!(sim)

        velsquaredsum = sum(x -> x^2, sim.vel)
        pressures[n] = (velsquaredsum + virial) / (sim.dim * sim.volume)
        temperatures[n] = velsquaredsum / (sim.dim * sim.num)
    end

    return potentials, temperatures, pressures
end

"""
Calculate the temperature (average kinetic energy of each degree of freedom) of the system
"""
function get_temperature(sim::MolSim)
    return sum(x -> x^2, sim.vel) / (sim.dim * sim.num)
end

"""
Calculate the potential of the system
"""
function get_potential(sim::MolSim)
    potential = 0.0

    for i in 1:(sim.num ÷ 2)
        londonterm, pauliterm, _ = _lennard_jones_terms(sim, i)
        potential += 4 * sum(pauliterm - londonterm)
    end

    # final "overlapping" shift for pairing particles
    if sim.num % 2 == 0
        londonterm, pauliterm, _ = _lennard_jones_terms(sim, sim.num ÷ 2 + 1)
        # half the potential since each pair is counted twice
        potential += 2 * sum(pauliterm - londonterm)
    end

    return potential
end

"""
Calculate the pressure of the system using the virial theorem
"""
function get_pressure(sim::MolSim)
    virial = 0.0

    for i in 1:(sim.num ÷ 2)
        londonterm, pauliterm, _ = _lennard_jones_terms(sim, i)
        virialterm = 48 * pauliterm - 24 * londonterm
        virial += sum(virialterm)
    end

    # final "overlapping" shift for pairing particles
    if sim.num % 2 == 0
        londonterm, pauliterm, _ = _lennard_jones_terms(sim, sim.num ÷ 2 + 1)
        # half the virial term since each pair is counted twice
        virialterm = 24 * pauliterm - 12 * londonterm
        virial += sum(virialterm)
    end

    return (sum(x -> x^2, sim.vel) + virial) / (sim.dim * sim.volume)
end

"""
Calculate the Van der Waals force for each particle, the Lennard-Jones potential and the
second term in the virial theorem formula for pressure times volume (sum of the inner
product of the forces and the respective distances)
"""
@inline function _calculate_vanderwaals(sim::MolSim{T}) where {T<:AbstractFloat}
    force = zeros(T, sim.dim, sim.num)
    potential = 0.0
    virial = 0.0

    for i in 1:(sim.num ÷ 2)
        londonterm, pauliterm, invdist = _lennard_jones_terms(sim, i)
        virialterm = 48 * pauliterm - 24 * londonterm
        addedforce = virialterm .* invdist

        force += addedforce
        force -= circshift(addedforce, (0, -i))
        potential += 4 * sum(pauliterm - londonterm)
        virial += sum(virialterm)
    end

    # final "overlapping" shift for pairing particles
    if sim.num % 2 == 0
        londonterm, pauliterm, invdist = _lennard_jones_terms(sim, sim.num ÷ 2 + 1)
        # half the virial term since each pair is counted twice
        virialterm = 24 * pauliterm - 12 * londonterm
        # only add the force in one direction, since the force of each pair is calculated
        # in both directions in this case and we don't need to use newton's third law
        force += virialterm .* invdist
        # half the potential since each pair is counted twice
        potential += 2 * sum(pauliterm - londonterm)
        virial += sum(virialterm)
    end

    return force, potential, virial
end

"""
Calculate the two terms of the lennard-jones potential and the inverse distance vetor
for pairs of particles shifted i columns from each other.
"""
@inline function _lennard_jones_terms(sim::MolSim{T}, i::Integer) where {T<:AbstractFloat}
    dist = sim.pos - circshift(sim.pos, (0, i))
    # periodic boundary conditions
    dist -= (dist .÷ (sim.sidelens / 2)) .* sim.sidelens

    distsquared = sum(x -> x^2, dist, dims=1)
    # term related to the london dispersion force, which is proportional to r^(-6)
    londonterm = distsquared .^ (-3)
    # term related to the pauli exclusion principle, which is proportional to r^(-12)
    pauliterm = distsquared .^ (-6)

    return londonterm, pauliterm, dist ./ distsquared
end

"""
Apply one step of the velocity verlet integration and also return the van der waals force
and the Lennard-Jones potential and the second term in the virial theorem formula for
pressure times volume (sum of forces times the respective distances)
"""
@inline function _velverlet!(sim::MolSim{T}, prevforce::Matrix{T}, stepsize::T
        ) where {T<:AbstractFloat}
    sim.pos += sim.vel * stepsize + prevforce * (stepsize * stepsize / 2)
    force, potential, virial = _calculate_vanderwaals(sim)
    sim.vel += (prevforce + force) * (stepsize / 2)

    return force, potential, virial
end

"""
Apply periodic boundary conditions on a simulation after each step
"""
@inline function _apply_boundary_conditions!(sim::MolSim)
    sim.pos = (sim.pos .+ sim.sidelens) .% sim.sidelens
end

end
