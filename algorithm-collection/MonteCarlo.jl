module MonteCarlo

export mcintegral

using Statistics

"""
Integrate a 1D real function using Monte Carlo integration with importance sampling

Does uniform sampling by default

# Arguments
- `func`: the function to be integrated
- `lo`: lower bound of the integral
- `up`: upper bound of the integral
- `samples`: the number of random samples used for integration
- `impfunc`: The importance sampling function.
    Constant function `x -> 1` by default (for uniform sampling).
- `impint`: the integral of the importance function in the integration interval. If equal to
    zero, this function will calculate the integral itself. `up - lo` by default
    (for uniform sampling).
- `distribution`: Applies this transformation function to the random numbers generated
    in the range [0, 1] to get the random samples used for integration. This must be
    a distribution that gives a probability density function proportional to impfunc
    to get the correct results. By default, samples are uniformly generated in the range
    [lo, up].

# Returns

The calculated integral and the standard error
"""
function mcintegral(func::Function, lo::Real, up::Real; samples::Integer = 10000,
        impfunc::Function = x -> 1, impint::Real = up - lo,
        distribution::Function = x -> muladd(x, up - lo, lo))
    if impint == 0
        impint = (up - lo) * mean(impfunc.(muladd.(rand(samples), up - lo, lo)))
    end

    intfunc(x) = func(x) / impfunc(x)
    samplevalues = intfunc.(distribution.(rand(samples)))
    avg = mean(samplevalues) 

    integral = impint * avg
    error = impint / √samples * std(samplevalues, mean=avg)

    return integral, error
end

"""
n-dimensional Monte Carlo integration with importance sampling

Does uniform sampling by default

# Arguments
- `func`: the function to be integrated
- `lo`: vector of lower bounds of the integral
- `up`: vector of upper bounds of the integral
- `samples`: the number of random samples used for integration
- `impfunc`: The importance sampling function.
    Constant function `(x...) -> 1` by default (for uniform sampling).
- `impint`: the integral of the importance function in the integration interval. If equal to
    zero, this function will calculate the integral itself. `up - lo` by default
    (for uniform sampling).
- `distribution`: Applies this transformation function to the random collection of numbers
    generated in the range [0, 1] to get the random samples used for integration. This must
    be a distribution that gives a probability density function proportional to impfunc
    to get the correct results. By default, samples are uniformly generated in the range
    [lo, up].

Please note that `length(lo)` must equal `length(up)` and all functions should also take
this many arguments as their input.

# Returns

The calculated integral and the standard error
"""
function mcintegral(func::Function, lo::Vector{<:Real}, up::Vector{<:Real};
        samples::Integer = 10000,
        impfunc::Function = (x...) -> 1, impint::Real = reduce(.*, up .- lo),
        distribution::Function = (x...) -> muladd.(x, up .- lo, lo))
    ndims = length(up)
    if impint == 0
        rndnums = rand(ndims, samples)
        samplepoints = [muladd.(x, up .- lo, lo) for x in eachcol(rndnums)]
        impint = (up - lo) * mean(impfunc(x...) for x in samplepoints)
    end

    rndnums = rand(ndims, samples)
    samplepoints = [distribution(x...) for x in eachcol(rndnums)]

    intfunc(x...) = func(x...) / impfunc(x...)
    samplevalues = [intfunc(x...) for x in samplepoints]
    avg = mean(samplevalues)

    integral = impint * avg
    error = impint / √samples * std(samplevalues, mean=avg)

    return integral, error
end

end
