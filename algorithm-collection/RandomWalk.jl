module RandomWalk

export randomwalk1d, randomwalk2d, rwtrap1d_lifetime, rwtrap1d_lifetime_exp, dlaline!

"""
1D Random Walk

# Arguments

-`time`: the number of steps of the random walk
-`p`: the probability of going right
-`walks`: the number of walks, defaults to `100*time`

# Returns

a vector containing the coordinates of each point and a vector counting in how many times
the walker landed on each point ath the end of the walk (only non-zero points are returned)
"""
function randomwalk1d(time::Integer, p::Float64; walks=100*time)
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
2D Random Walk

# Arguments

-`time::Integer`: the number of steps of the random walk
-`walks::Integer`: the number of walks, defaults to the value of `time`

# Returns

two tuples of two integers giving the x and y limits of the points and
an integer matrix counting the number of times the walker landed on each point.
"""
function randomwalk2d(time::Integer; walks::Integer=100*time)
    x, y = collect(Integer, -time:time), collect(Integer, -time:time)
    grid = zeros(Integer, 2 * time + 1, 2 * time + 1)
    steps = ((0, 1), (1, 0), (0, -1), (-1, 0))

    for _ in 1:walks
        xpos, ypos = time + 1, time + 1
        for _ in 1:time
            step = steps[rand(1:4)]
            xpos += step[1]
            ypos += step[2]
        end
        grid[xpos, ypos] += 1
    end

    return randomwalk2dcrop(x, y, grid)
end

"""
crop the output of `randomwalk2d`
"""
@inline function randomwalk2dcrop(x::Vector{<:Integer}, y::Vector{<:Integer},
        grid::Matrix{<:Integer})
    height, width = size(grid)
    bottom = left = 1
    top = height
    right = width

    while all(==(0), grid[top, :])
        top -= 1
    end
    while all(==(0), grid[bottom, :])
        bottom += 1
    end
    while all(==(0), grid[:, left])
        left += 1
    end
    while all(==(0), grid[:, right])
        right -= 1
    end

    return x[left:right], y[bottom:top], grid[bottom:top, left:right]
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

"""
Applies diffusion-limited aggregation on a 2D surface, assuming the "seed" is a line

the line is at the bottom and new particles appear from the top of the pile

# Arguments

- `surface::BitMatrix`: The simulation surface. Particles occupy points that are true.
- `particles::Integer`: the number of particles (walkers) simulated
- `escapethresh::Integer`: if the particle gets this much higher from the pile, it will
    "escape". It will be removed from the simulation and a new particle will be generated
    in its place. Defaults to 5.

# Returns
the `surface`
"""
function dlaline!(surface::BitMatrix, particles::Integer; escapethresh::Integer=5)
    height, width = size(surface)
    maxheight = height
    while true ∉ surface[maxheight - 1, :]
        maxheight -= 1
    end
    steps = ((0, 1), (1, 0), (0, -1), (-1, 0))

    while particles > 0
        ypos, xpos = maxheight, rand(1:width)

        while true
            xprev, yprev = xpos, ypos
            step = steps[rand(1:4)]
            ypos += step[1]
            xpos += step[2]
            # periodic boundary conditions (if xpos = width + 1, then it will be shifted
            # left by width, and if xpos = 0, it will be shifted right by width)
            xpos += (width + 1 - 2xpos) ÷ (width + 1) * width

            if ypos > min(maxheight + escapethresh, height)
                break
            elseif surface[ypos, xpos]
                surface[yprev, xprev] = true
                particles -= 1
                if yprev > maxheight
                    maxheight = yprev
                end
                break
            end
        end
    end
    return surface # for convenience
end

end
