module Percolation

export dfspercolate, dfspercolate!, colorpercolate, hkpercolate

using DataStructures

"""
Find out whether or not percolation is possible in a 2D grid using depth-first search

Percolation is assumed to occur from top to bottom.

# Argument

`grid::Matrix{Bool}`: Represents the grid cells; True if porous (part of the path;
    can pass through) and False if blocked (can't pass through; no path passes through)

# Returns

boolean indicating whether percolation is possible
"""
function dfspercolate(grid::BitMatrix)
    height, width = size(grid)

    undiscovered = BitMatrix(undef, height + 2, width + 2)
    # "pad" the array to avoid bound checking for neigboring
    undiscovered[:, 1] .= false
    undiscovered[1, :] .= false
    undiscovered[end, :] .= false
    undiscovered[:, end] .= false
    # insert grid into undiscovered array
    undiscovered[2:end-1, 2:end-1] = grid

    stack = Stack{Tuple{Integer, Integer}}()
    for root in 2:width+1
        if undiscovered[2, root] # top row
            # search this path
            push!(stack, (2, root))
            if dfs!(height + 1, stack, undiscovered)
                return true
            end
        end
    end
    return false
end

"""
Depth-first search algorithm adopted for the dfspercolate function
"""
@inline function dfs!(destination::Integer, stack::Stack, undiscovered::BitMatrix)
    while !isempty(stack)
        vertex = pop!(stack)
        undiscovered[vertex...] = false

        # check if destination is reached
        if vertex[1] == destination
            return true
        end

        # check neighboring vertices
        for step in ((-1, 0), (0, -1), (0, 1), (1, 0))
            neighbor = vertex .+ step
            if undiscovered[neighbor...]
                push!(stack, neighbor)
            end
        end
    end
    return false
end

"""
Find out whether or not percolation is possible in a 2D grid using depth-first search

Percolation is assumed to occur from top to bottom. this version marks points visited by
the dfs in the visited

# Arguments

- `grid::Matrix{Bool}`: Represents the grid cells; True if porous (part of the path;
    can pass through) and False if blocked (can't pass through; no path passes through)
- `visited::BitMatrix`: whether each point is visited by the dfs

# Returns

boolean indicating whether percolation is possible
"""
function dfspercolate!(grid::BitMatrix, visited::BitMatrix)
    height, width = size(grid)

    undiscovered = BitMatrix(undef, height + 2, width + 2)
    # "pad" the array to avoid bound checking for neigboring
    undiscovered[:, 1] .= false
    undiscovered[1, :] .= false
    undiscovered[end, :] .= false
    undiscovered[:, end] .= false
    # insert grid into undiscovered array
    undiscovered[2:end-1, 2:end-1] = grid

    stack = Stack{Tuple{Integer, Integer}}()
    for root in 2:width+1
        if undiscovered[2, root] # top row
            # search this path
            push!(stack, (2, root))
            if dfs!(height + 1, stack, undiscovered, visited)
                return true
            end
        end
    end
    return false
end

"""
Depth-first search algorithm adopted for the dfspercolate function

this version modifies the `visited` matrix
"""
@inline function dfs!(destination::Integer, stack::Stack, undiscovered::BitMatrix,
        visited::BitMatrix)
    while !isempty(stack)
        vertex = pop!(stack)
        undiscovered[vertex...] = false
        visited[(vertex .- (1, 1))...] = true

        # check if destination is reached
        if vertex[1] == destination
            return true
        end

        # check neighboring vertices
        for step in ((-1, 0), (0, -1), (0, 1), (1, 0))
            neighbor = vertex .+ step
            if undiscovered[neighbor...]
                push!(stack, neighbor)
            end
        end
    end
    return false
end

"""
labels the clusters in a 2D grid for solving the percolation problem

Uses and inefficient O((length * width)^2) coloring algorithm.
Percolation is assumed to occur from top to bottom.

# Argument

`grid::Matrix{Bool}`: Represents the grid cells; True if porous (part of the path;
    can pass through) and False if blocked (can't pass through; no path passes through)

# Returns

a boolean indicating whether percolation is possible and a matrix "colored" with integers.
Points with value 1 are connected to the starting row.
"""
function colorpercolate(grid::BitMatrix)
    height, width = size(grid)
    colors = zeros(Integer, height + 1, width)
    colors[1, :] .= 1
    counter = 2
    
    for row in 2:height+1
        for col in 1:width
            if grid[row - 1, col]
                neighbors = neighborhood(colors, row, col)
                nonzero = filter(n -> colors[n...] != 0, neighbors)
                if length(nonzero) == 0
                    colors[row, col] = counter
                    counter += 1
                elseif length(nonzero) == 1
                    colors[row, col] = colors[nonzero[1]...]
                else
                    minneighbor = minimum([colors[n...] for n in nonzero])
                    colors[row, col] = minneighbor
                    for neighbor in nonzero
                        if colors[neighbor...] != minneighbor
                            colors[colors .== colors[neighbor...]] .= minneighbor
                        end
                    end
                end
            end
        end
    end
    return any(x -> x == 1, colors[end, :]), colors[2:end, :]
end

"""
Retruns a vector of tuples of the neighbors of the point at [row, col]
"""
@inline function neighborhood(grid::Matrix, row::Integer, col::Integer)
    neighbors = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
    return filter(n -> checkbounds(Bool, grid, n...), neighbors)
end

"""
Uses the Hoshen-Kopelman algorithm to label the clusters in a 2D grid for percolation

Percolation is assumed to occur from top to bottom.

# Arguments

- `grid::Matrix{Bool}`: Represents the grid cells; True if porous (part of the path;
    can pass through) and False if blocked (can't pass through; no path passes through)
- `color::Bool` (default = `false`): whether to color the grid after finishing the algorithm

# Returns

[a boolean indicating whether percolation is possible or, if qinfty is true,
the sum of the sizes of the open connecting clusters (integer)],
an integer matrix where clusters, are marked by unique integers, and the cluster lengths.
"""
function hkpercolate(grid::BitMatrix; color::Bool=false, qinfty::Bool=false)
    height, width = size(grid)
    labels = zeros(Integer, height, width)
    list = Vector{Integer}()
    clusters = Vector{Integer}()

    # first row, special case because of top boundary
    hkfirstrow!(grid, labels, list, clusters)

    for row in 2:height
        # first column, special case because of left boundary
        if grid[row, 1]
            hkfirstcol!(labels, list, clusters, row)
        end

        # "middle" points, with appropriate neighbors
        for col in 2:width
            if grid[row, col]
                hkiter!(labels, list, clusters, row, col)
            end
        end
    end

    if color
        hkcolor!(labels, list)
    end
    if qinfty
        return hkresult(labels, list, clusters), labels, clusters
    else
        return hkbool(labels, list), labels, clusters
    end
end

"""
Helper function for `hkpercolate`; Adds new label.
"""
@inline function hknew!(labels::Matrix{<:Integer}, list::Vector{<:Integer},
        clusters::Vector{<:Integer}, row::Integer, col::Integer)
    labels[row, col] = length(list) + 1
    push!(list, labels[row, col])
    push!(clusters, 0)
end

"""
Helper function for `hkpercolate`; Applies the Hoshen-Kopelman algorithm to the first row.
"""
@inline function hkfirstrow!(grid::BitMatrix, labels::Matrix{<:Integer},
        list::Vector{<:Integer}, clusters::Vector{<:Integer})
    width = size(labels)[2]
    if grid[1, 1]
        hknew!(labels, list, clusters, 1, 1)
        clusters[labels[1, 1]] += 1
    end
    for col in 2:width
        if grid[1, col]
            if labels[1, col - 1] != 0
                labels[1, col] = hkfind!(list, labels[1, col - 1])
            else
                hknew!(labels, list, clusters, 1, col)
            end
            clusters[labels[1, col]] += 1
        end
    end
end

"""
Helper function for `hkpercolate`; Applies the Hoshen-Kopelman algorithm to the first column.
"""
@inline function hkfirstcol!(labels::Matrix{<:Integer}, list::Vector{<:Integer},
        clusters::Vector{<:Integer}, row::Integer)
    if labels[row - 1, 1] != 0
        labels[row, 1] = hkfind!(list, labels[row - 1, 1])
    else
        hknew!(labels, list, clusters, row, 1)
    end
    clusters[labels[row, 1]] += 1
end

"""
Helper function for `hkpercolate`; Applies one iteration of the Hoshen-Kopelman algorithm.
"""
@inline function hkiter!(labels::Matrix{<:Integer}, list::Vector{<:Integer},
        clusters::Vector{<:Integer}, row::Integer, col::Integer)
    above, left = labels[row - 1, col], labels[row, col - 1]
    if above == 0
        if left == 0
            hknew!(labels, list, clusters, row, col)
        else
            labels[row, col] = hkfind!(list, left)
        end
    elseif left == 0
        labels[row, col] = hkfind!(list, above)

    else
        labels[row, col] = hkunion!(list, clusters, above, left)
    end
    clusters[labels[row, col]] += 1
end

"""
Union function of the union-find algorithm used by the Hoshen-Kopelman algorithm
"""
@inline function hkunion!(list::Vector{<:Integer}, clusters::Vector{<:Integer},
        x::Integer, y::Integer)
    root = hkfind!(list, x)
    dest = hkfind!(list, y)
    if dest != root
        list[dest] = root
        clusters[root] += clusters[dest]
        clusters[dest] = 0
    end
    return root
end

"""
Find function of the union-find algorithm used by the Hoshen-Kopelman algorithm
"""
@inline function hkfind!(list::Vector{<:Integer}, x::Integer)
    y = x;
    while list[y] != y
        y = list[y];
    end
    while list[x] != x
        z = list[x];
        list[x] = y;
        x = z;
    end
    return y;
end

"""
Same as hkfind!, but doesn't update labels
"""
@inline function hkfind(list::Vector{<:Integer}, x::Integer)
    while x != list[x]
        x = list[x];
    end
    return x
end

"""
Color resultant grid of the Hoshen-Kopelman algorithm appropriately
"""
@inline function hkcolor!(labels::Matrix{<:Integer}, list::Vector{<:Integer})
    width, height = size(labels)
    for row in 1:height
        for col in 1:width
            if labels[row, col] != 0
                labels[row, col] = hkfind(list, labels[row, col])
            end
        end
    end
end

"""
Find the sum of the sizes of the open connecting clusters (integer)
"""
@inline function hkresult(labels::Matrix{<:Integer}, list::Vector{<:Integer},
        clusters::Vector{<:Integer})
    sumsize = 0
    # to avoid over-counting the clusters, save the label of each cluster that is counted
    counted = []

    for root in labels[1, :]
        if root > 0
            for dest in labels[end, :]
                if dest > 0
                    toplabel = hkfind(list, root)
                    bottomlabel = hkfind(list, dest)
                    if toplabel == bottomlabel && !(toplabel in counted)
                        sumsize += clusters[toplabel]
                        append!(counted, toplabel)
                    end
                end
            end
        end
    end
    return sumsize
end

"""
Find out whether percolation is possible or not in hkpercolate output
"""
@inline function hkbool(labels::Matrix{<:Integer}, list::Vector{<:Integer})
    for root in labels[1, :]
        if root > 0
            for dest in labels[end, :]
                if dest > 0 && hkfind(list, dest) == hkfind(list, root)
                    return true
                end
            end
        end
    end
    return false
end

end
