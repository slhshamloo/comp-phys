using Statistics

"""
Calculate the autocorrelation of `data` at the given distance `dist`
"""
function autocorrelation(data::Vector{<:Real}, dist::Integer)
    xᵢ = data[1:(length(data) - dist)]
    xᵢ₊ⱼ = data[(1 + dist):end]
    xᵢxᵢ₊ⱼavg = mean(xᵢ .* xᵢ₊ⱼ)
    return (xᵢxᵢ₊ⱼavg - mean(xᵢ) * mean(xᵢ₊ⱼ)) / var(data)
end
