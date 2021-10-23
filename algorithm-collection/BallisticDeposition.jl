module BallisticDeposition

export bdeposition!, bdfull!, bdiso!, bdisofull!, bdrelax!, bdheights

"""
Apply ballistic deposition to a 1D surface with periodic boundary conditions

# Arguments

- `surface::Vector{Integer}`: the 1D array representing the surface
- `time::Integer`: the number of time steps (or the number of deposited particles)

# Returns

the surface heights (Vector) after applying the ballistic deposition
"""
function bdeposition!(surface::Vector{<:Integer}, time::Integer)
    len = length(surface)
    starting_indecies = rand(1:len, time)

    for i in starting_indecies
        # finding the left and right indecies with periodic boundary conditions
        left = i - 1 + ((len + 1 - i) ÷ len) * len
        right = i + 1 - (i ÷ len) * len
        
        # the particle will "stick" to the largest column
        surface[i] = max(surface[left], surface[i] + 1, surface[right])
    end
    
    return surface
end

"""
Apply ballistic deposition to a 1D surface with periodic boundary conditions

# Arguments

- `surface::Vector{Integer}`: the 1D array representing the surface
- `time::Integer`: the number of time steps (or the number of deposited particles)

# Returns

boolean matrix containing the state of each cell (full / empty)
"""
function bdfull!(surface::Vector{<:Integer}, time::Integer)
    len = length(surface)
    starting_indecies = rand(1:len, time)
    grid = falses(time + maximum(surface), len)

    for i in starting_indecies
        # finding the left and right indecies with periodic boundary conditions
        left = i - 1 + ((len + 1 - i) ÷ len) * len
        right = i + 1 - (i ÷ len) * len
        
        # the particle will "stick" to the largest column
        surface[i] = max(surface[left], surface[i] + 1, surface[right])
        grid[surface[i], i] = true
    end
    
    return grid[1:(maximum(surface)), :]
end

"""
Apply ballistic deposition for an isolated "tree" of particles on a 1D surface

# Arguments

- `surface::Vector{Integer}`: The 1D array representing the surface. Must only contain one
    connected tree of particles.
- `time::Integer`: the number of time steps (or the number of deposited particles)

# Returns

the surface heights (Vector) after applying the ballistic deposition.
"""
function bdiso!(surface::Vector{<:Integer}, time::Integer)
    left, right = findleftright(surface)

    for _ in 1:time
        # The new particles can only be dropped on neighbors of the current points
        index = rand(left:right)

        # the particle will "stick" to the largest column
        surface[index] = max(surface[index - 1], surface[index] + 1, surface[index + 1])
        
        # expand the domain of dropping the particle
        if index == right
            right += 1
        elseif index == left
            left -= 1
        end
    end
    
    return surface
end

"""
Apply ballistic deposition for an isolated "tree" of particles on a 1D surface

# Arguments

- `surface::Vector{Integer}`: The 1D array representing the surface. Must only contain one
    connected tree of particles.
- `time::Integer`: the number of time steps (or the number of deposited particles)

# Returns

boolean matrix containing the state of each cell (full / empty)
"""
function bdisofull!(surface::Vector{<:Integer}, time::Integer)
    len = length(surface)
    grid = falses(time + maximum(surface), len)
    left, right = findleftright(surface)

    for _ in 1:time
        # The new particles can only be dropped on neighbors of the current points
        index = rand(left:right)

        # the particle will "stick" to the largest column
        surface[index] = max(surface[index - 1], surface[index] + 1, surface[index + 1])
        grid[surface[index], index] = true

        # expand the domain of dropping the particle
        if index == right
            right += 1
        elseif index == left
            left -= 1
        end
    end
    
    return grid[1:(maximum(surface)), :]
end

"""
Find left and right of the isolated tree in ballistic deposition (for bdiso functions)
"""
function findleftright(surface)
    leftindex = 1
    while surface[leftindex + 1] == 0
        leftindex += 1
    end

    rightindex = length(surface)
    while surface[rightindex - 1] == 0
        rightindex -= 1
    end

    return leftindex, rightindex
end

"""
Apply ballistic deposition with relaxation to a 1D surface with periodic boundary conditions

# Arguments

- `surface::Vector{Integer}`: the 1D array representing the surface
- `time::Integer`: the number of time steps (or the number of deposited particles)

# Returns

the surface array (vector) after applying the ballistic deposition
"""
function bdrelax!(surface::Vector{<:Integer}, time::Integer)
    len = length(surface)
    starting_indecies = rand(1:len, time)

    for i in starting_indecies
        # finding the left and right indecies with periodic boundary conditions
        left = i - 1 + ((len + 1 - i) ÷ len) * len
        right = i + 1 - (i ÷ len) * len
        
        # add one particle to the position with the minimum height
        # choose from left, i, and right
        if surface[i] > surface[left]
            if surface[left] > surface[right]
                surface[right] += 1
            else
                surface[left] += 1
            end
        elseif surface[i] > surface[right]
            surface[right] += 1
        else
            surface[i] += 1
        end
    end
    
    return surface
end

"""
Generate the heights at the given times for the given ballistic deposition

# Arguments

- `bdfunc!::Function`: a function that applies a ballistic deposition to a surface vector
    for a given time and returns the surface (it doesn't matter if the surface itself is
    modified or not)
- `len::Integer`: the length of the 1D surface
- `times::Vector{<:Integer}`: the times of each recorded height

# Returns

A matrix containing the array heights for the given times; Each row corresponds to one time
"""
function bdheights(bdfunc::Function, len::Integer, times::Vector{<:Integer})
    heights = Matrix{Integer}(undef, len, length(times))

    heights[:, 1] = bdfunc(zeros(Integer, len), times[1])
    for i in 2:length(times)
        heights[:, i] = bdfunc(heights[:, i - 1], times[i] - times[i - 1])
    end

    return heights
end

end
