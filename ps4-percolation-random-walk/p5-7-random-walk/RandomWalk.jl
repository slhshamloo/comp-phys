module RandomWalk

export randomwalk1d, rwtrap1d_lifetime, rwtrap1d_lifetime_exp

"""
1D Random Walk

# Arguments

-`time::Integer`: the number of steps of the random walk
-`p::Float64`: the probability of going right
-`walks::Integer`: the number of walks, defaults to `100*time`

# Returns

a vector containing the coordinates of each point and a vector counting how many times
the walker landed on each point ath the end of the walk (only non-zero points are returned)
"""
function randomwalk1d(time::Integer, p::Float64; walks::Integer=100*time)
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

"""
Calculate the average lifetime of a 1D random walker inside trapping boundaries

# Arguments

-`len::Integer`: The length of the path. The coordinates are in the range [-len÷2, len÷2]
-`initpos::Integer`: the initial position of the walker
-`walks::Integer`: the number of walks, defaults to the value of `time`
-`maxtime::Integer`: The maximum number of steps before the walker is abandoned;
    in case the number of steps exceeds `maxtime`, the lifetime is taken to be `maxtime`.
    defaults to `100*len`.

# Returns

The average lifetime, which is the average time passed before the walker gets trapped
in the boundaries of the path
"""
function rwtrap1d_lifetime(len::Integer, initpos::Integer, walks::Integer;
        maxtime::Integer=100*len)
    lifetime_sum = 0

    for _ in 1:walks
        pos = initpos
        time = 0

        while time < maxtime
            time += 1
            if rand() <= 0.5
                pos += 1
                if pos > len ÷ 2
                    break
                end
            else
                pos -= 1
                if pos < -len ÷ 2
                    break
                end
            end
        end

        lifetime_sum += time
    end

    lifetime_avg = lifetime_sum / walks
    return lifetime_avg
end

"""
Calculate the expected lifetime of a 1D random walker inside trapping boundaries

# Arguments

-`len::Integer`: The length of the path. The coordinates are in the range [-len÷2, len÷2]
-`initpos::Integer`: the position of the boundaries (traps)
-`maxtime::Integer`: The maximum number of steps before the walker is abandoned;
    in case the number of steps exceeds `maxtime`, the lifetime is taken to be `maxtime`.

# Returns

The average lifetime, which is the average time passed before the walker gets trapped
in the boundaries of the path
"""
function rwtrap1d_lifetime_exp(len::Integer, initpos::Integer; maxtime::Integer=100*len)
    lifetime_avg = 0

    probs = zeros(len + 3)
    probs[initpos + len ÷ 2 + 2] = 1

    for _ in 1:maxtime
        newprobs = zeros(len + 3)
        for i in 2:len+2
            newprobs[i - 1] += probs[i] * 0.5
            newprobs[i + 1] += probs[i] * 0.5
        end
        probs = newprobs

        lifetime_avg += sum(probs[2:end-1])
    end

    return lifetime_avg
end

end
