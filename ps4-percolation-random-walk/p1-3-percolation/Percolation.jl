module Percolation

export clustgyration, hklabel, hkmaxclust!, genclust, clustfractal

using DataStructures

"""
Finds the radius of gyration for the largest closed cluster in the percolation grid

Percolation is assumed to occur from top to bottom. Uses the Hoshen-Kopelman algorithm.

# Argument

`grid::Matrix{Bool}`: Represents the grid cells; True if porous (part of the path;
    can pass through) and False if blocked (can't pass through; no path passes through)

# Returns

The gyration radius of the largest closed cluster (the largest cluster that doesn't connect
the top and bottom of the grid)
"""
function clustgyration(grid::BitMatrix)
    labels, list, clusters = hklabel(grid)
    if length(clusters) == 0
        return 0
    end

    maxlabel = hkmaxclust!(labels, list, clusters)
    if maxlabel == 0
        return 0
    end

    maxclust = Tuple.(findall(x -> x != 0 && hkfind(list, x) == maxlabel, labels))
    mass = length(maxclust)
    if mass == 0
        return 0
    end

    centerofmass = reduce(.+, maxclust) ./ mass
    return √(sum(pos -> reduce(.+, (pos .- centerofmass).^2),
        maxclust) / mass)
end

"""
Uses the Hoshen-Kopelman algorithm to label the grid

returns the labels, label mapping list, and cluster lengths
"""
@inline function hklabel(grid::BitMatrix)
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

    return labels, list, clusters
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
Finds the label of the largest closed cluster (a.k.a. finite cluster).

Also, sets the size of open clusters (a.k.a. infinite cluster) to zero, hence the !
"""
@inline function hkmaxclust!(labels::Matrix{Integer}, list::Vector{Integer},
        clusters::Vector{Integer})
    for root in labels[1, :]
        if root > 0
            for dest in labels[end, :]
                if dest > 0
                    toplabel = hkfind(list, root)
                    bottomlabel = hkfind(list, dest)
                    if toplabel == bottomlabel
                        clusters[toplabel] = 0
                    end
                end
            end
        end
    end

    if maximum(clusters) > 0
        return argmax(clusters)
    else
        return 0
    end
end

"""
Uses a depth-first search algorithm to generate a cluster with the given site probability

# Arguments

- `probability`: the probability a site is generated when growing the cluster
- `maxsize::Integer` (optional, defaults to 10000): the maximum size of the cluster
    (breaks the loop after the cluster is grown to this size)

# Returns

an integer matrix (8-bit) with 0 where no site exists, 1 where a cell is generated, and 2
where the growth is stopped
"""
function genclust(probability::Float64; maxsize::Integer=10000, crop::Bool=true)
    cluster = zeros(Int8, maxsize, maxsize)
    # create "fences" to stop the cluster form reaching the bounds of the matrix
    cluster[1, :] .= 2
    cluster[:, 1] .= 2
    cluster[end, :] .= 2
    cluster[:, end] .= 2

    stack = Stack{Tuple{Integer, Integer}}()
    seed = (maxsize÷2, maxsize÷2)
    push!(stack, seed)
    cluster[seed...] = 1

    clustdfs!(stack, cluster, probability, maxsize)

    if crop
        return cropclust!(cluster)
    else
        return cluster
    end
end

"""
The depth-first search algorithm used for `genclust`
"""
@inline function clustdfs!(stack::Stack, cluster::Matrix{<:Integer},
        probability::Float64, maxsize::Integer)
    size = 1
    while !isempty(stack) && size < maxsize
        site = pop!(stack)
        for step in ((-1, 0), (0, -1), (0, 1), (1, 0))
            neighbor = site .+ step
            if cluster[neighbor...] == 0
                if rand() <= probability
                    cluster[neighbor...] = 1
                    push!(stack, neighbor)
                    size += 1
                else
                    cluster[neighbor...] = 2
                end
            end
        end
    end
end

"""
finds the mass and radius of gyration of the cluster from the output of `genclust`
"""
function clustfractal(cluster::Matrix{Int8})
    clust = Tuple.(findall(x -> x == 1, cluster))
    mass = length(clust)
    if mass == 0
        return 0
    end

    centerofmass = reduce(.+, clust) ./ mass
    return mass, √(sum(pos -> reduce(.+, (pos .- centerofmass).^2), clust) / mass)
end

"""
crop the `genclust` function output to remove excess zeros
"""
@inline function cropclust!(cluster::Matrix{<:Integer})
    height, width = size(cluster)
    top = bottom = height ÷ 2
    left = right = width ÷ 2

    while 1 in cluster[top + 1, 2:end-1] || 2 in cluster[top + 1, 2:end-1]
        top += 1
    end
    while 1 in cluster[bottom - 1, 2:end-1] || 2 in cluster[bottom - 1, 2:end-1]
        bottom -= 1
    end
    while 1 in cluster[2:end-1, left - 1] || 2 in cluster[2:end-1, left - 1]
        left -= 1
    end
    while 1 in cluster[2:end-1, right + 1] || 2 in cluster[2:end-1, right + 1]
        right += 1
    end

    return cluster[bottom:top, left:right]
end

end
