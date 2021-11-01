"""
Count every possible self-avoiding walk of length `n`

Simple wrapper for sawrecurse
"""
function countsaw(n::Integer)
    return sawrecurse!(n, falses(2n + 1, 2n + 1), n + 1, n + 1)
end

let steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
    global sawrecurse!
    """
    Recursively find all self-avoiding walks of length `n` from the position `grid[x, y]`

    The grid is `true` where the path passes through the node, and false otherwise
    """
    function sawrecurse!(n::Integer, grid::BitMatrix, x::Integer, y::Integer)
        if n == 0
            return 1
        else
            # enter path
            grid[x, y] = true
            count = 0

            for step in steps
                xnew, ynew = x + step[1], y + step[2]
                if !grid[xnew, ynew] # don't cross path
                    count += sawrecurse!(n - 1, grid, xnew, ynew)
                end
            end

            # backtrack
            grid[x, y] = false
            return count
        end
    end
end
