from numpy import ndarray
from fractal import fractal_random
from typing import Sequence


def barnsley_fern(range_x: Sequence[float], range_y: Sequence[float],
                  steps: int = 20, samples: int = 20000
                  ) -> tuple[ndarray, ndarray]:
    """Generates a Barnsley Fern

    Args:
        init_x: x coordinates of the starting points
        init_y: y coordinates of the starting points
        steps: the number of times the mapping is applied

    Returns:
        two lists containing the x and y coordinates of the fractal,
        in correct order for drawing the constituent lines
    """
    height = range_y[1] - range_y[0]
    return fractal_random(
        range_x, range_y,
        (((0, 0), (0, 0.16)), ((0.85, 0.04), (-0.04, 0.85)),
         ((0.2, -0.26), (0.23, 0.22)), ((-0.15, 0.28), (0.26, 0.24))),
        ((0, 0), (0, 1.6 * height), (0, 1.6 * height), (0, 0.44 * height)),
        steps, samples, (0.01, 0.85, 0.07, 0.07))
