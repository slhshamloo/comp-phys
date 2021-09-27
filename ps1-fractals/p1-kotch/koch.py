from math import pi
from typing import Sequence
from fractal import fractal_point


def koch(init_x: Sequence[float], init_y: Sequence[float],
         steps: int = 6) -> tuple[list[float], list[float]]:
    """Draws the koch curve on the lines between the given points

    Args:
        init_x: x coordinates of the starting points
        init_y: y coordinates of the starting points
        steps: the number of times the mapping is applied

    Returns:
        two lists containing the x and y coordinates of the koch curve
    """
    return fractal_point(init_x, init_y, ((1 / 3, 1 / 3),) * 3,
                         (0, pi / 3, 0), ((0, 0), (1, 1), (1, 1)), steps)
