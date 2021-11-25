module Metropolis

export metropolis, metropolis_uniform

using Distributions

"""
Generate a series of random samples from a given probability distribution
using the metropolis algorithm

# Arguments
- `distribution`: the target probability distribution
- `step`: the probability distribution of the steps
- `samples`: the number of samples
- `init`: the initial value of the series (starting point of the "walker")

# Returns

the series of randome samples from the distribution and the acceptance rate
"""
function metropolis(distribution::Function, step::Function, samples::Integer;
        init::Real=0.0)
    series = Vector{Float64}(undef, samples)

    position = init
    probability = distribution(position)
    series[1] = position

    acceptcount = 0
    for i in 2:samples
        newposition = position + step(position)
        newprobablity = distribution(newposition)

        if newprobablity / probability > 1 || rand() <= newprobablity / probability
            position, probability = newposition, newprobablity
            acceptcount += 1
        end

        series[i] = position
    end

    return series, acceptcount / (samples - 1)
end

"""
Metropolis algorithm with uniform step probability between `-stepsize` and `+stepsize`
"""
function metropolis_uniform(distribution::Function, stepsize::Real, samples::Integer;
        init::Real=0.0)
    return metropolis(distribution, x -> rand(Uniform(-stepsize, stepsize)), samples,
        init=init)
end

end
