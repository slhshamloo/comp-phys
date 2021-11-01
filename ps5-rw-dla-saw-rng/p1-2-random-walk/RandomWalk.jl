module RandomWalk

export randomwalk2d, dlaline!

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