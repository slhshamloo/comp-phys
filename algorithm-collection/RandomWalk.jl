module RandomWalk

export randomwalk, dlaline!

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
