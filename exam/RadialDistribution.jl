"""
in the `positions` matrix, each column represents one particle and each row is one of
the coordinates (x, y, z, ...). If the input coordinates weren't in this format, it's
trivial to convert them to the desired matrix. A particle is counted if it's between `r`
and `r + dr` away from another particle.
"""
function radial_distribution(positions::Matrix{<:Real}, r::Real, dr::Real)
    inrange = 0
    for pos in eachcol(positions)
        distances = .√(sum(x -> x^2, positions .- pos, dims=1))
        inrange += count(x -> ((x <= r + dr) && (x >= r)), distances)
    end
    _, particle_count = size(positions)
    # average over all particles, then divide by the number of particles minus one because
    # the particle is never in range of "itself"
    return inrange / particle_count / (particle_count - 1)
end