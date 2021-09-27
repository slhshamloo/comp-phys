from numpy import ndarray
from fractal import fractal_random_scalerot


def sierpinski_random(range_x: tuple[float, float],
                      range_y: tuple[float, float],
                      steps: int = 10, samples: int=10000
                      ) -> tuple[ndarray, ndarray]:
    """Generates the points of a sierpinski triangle randomly

    Applies sierpinski triangle mappings randomly to randomly sampled points
    in the given range to approximately make a sierpinski triangle.

    Args:
        range_x: range of the sampling points in the x axis
        range_y: range of the sampling points in the y axis
        samples: the number of randomly sampled points
            for generating the fractal
        steps: the number of times the fractal mapping is applied

    Returns:
        two numpy arrays containing the x and y coordinates
        of the generated fractal's points
    """
    return fractal_random_scalerot(
        range_x, range_y, ((0.5, 0.5),) * 3, (0,) * 3,
            ((0, 0), (0.25 * (range_x[1] - range_x[0]),
                      0.5 * (range_y[1] - range_y[0])),
             (0.5 * (range_x[1] - range_x[0]), 0)), steps, samples)
