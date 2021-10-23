module RandomWalk

export randomwalk

"""
1D Random Walk

# Arguments

-`time`: the number of steps of the random walk
-`p`: the probability of going right
-`walks`: the number of walks

# Returns

a vector containing the coordinates of each point and a vector counting in how many times
the walker landed on each point ath the end of the walk (only non-zero points are returned)
"""
function randomwalk1d(time::Integer, p::Float64; walks=time)
    path = zeros(2 * time + 1)
    coordinates = collect(Integer, -time:time)

    for _ in 1:walks
        pos = time + 1
        for _ in 1:time
            if rand() <= p
                pos += 1
            else
                pos -= 1
            end
        end
        path[pos] += 1
    end

    nonzero = findall(x -> x != 0, path)
    return coordinates[nonzero], path[nonzero]
end

end
