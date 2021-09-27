from typing import Sequence
from fractal import fractal_shape


def sierpinski(init_x: Sequence[float], init_y: Sequence[float],
               steps: int = 8) -> tuple[list[float], list[float]]:
    """Generates a Sierpinski Triangle

    Args:
        init_x: x coordinates of the starting points
        init_y: y coordinates of the starting points
        steps: the number of times the mapping is applied

    Returns:
        two lists containing the x and y coordinates of the fractal,
        in correct order for drawing the constituent polygons
    """
    return fractal_shape(init_x, init_y, ((0.5, 0.5),) * 3, (0,) * 3,
                         ((0, 0), (0.5 * (init_x[1] - init_x[0]),
                                   0.5 * (init_y[1] - init_y[0])),
                          (0.5 * (init_x[-1] - init_x[0]), 0)), steps)
