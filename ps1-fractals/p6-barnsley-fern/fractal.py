import numpy as np
from typing import Optional, Sequence, Union


def fractal_random(
        range_x: Sequence[Union[np.ndarray, tuple[float, float]]],
        range_y: Sequence[Union[np.ndarray, tuple[float, float]]],
        transformations: Sequence[Union[np.ndarray, tuple[
            tuple[float, float], tuple[float, float]]]],
        shifts: Sequence[Union[np.ndarray, tuple[float, float]]],
        steps: int = 10, samples: int = 100000,
        weights: Optional[Sequence[float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Generates the points of a fractal by randomly applying mappings

    The mappings consist of a transformation (transform) and a shift.
    (xf, yf) = (transform)(xi, yi) + shift
    transform = ((a, b), (c, d))
    shift = (shiftx, shifty)

    Args:
        range_x: range of the sampling points in the x axis
        range_y: range of the sampling points in the y axis
        transforms: transformations applied to points before shifting in
            each mapping
        shifts: the shifts in coordinates of the shape in each mapping
            of the fractal
        steps: the number of times the fractal mapping is applied
        samples: the number of randomly sampled points
            for generating the fractal
        weights: The weight of each mapping in random selection.
            If None is given, the weights of every mapping will be equal.


    Returns:
        two numpy arrays containing the x and y coordinates
        of the generated fractal's points
    """
    map_count = len(transformations)
    if not isinstance(transformations[0], np.ndarray):
        transformations = [
            np.array(transformations[i]) for i in range(map_count)]

    x = np.random.rand(samples) * (range_x[1] - range_x[0]) + range_x[0]
    y = np.random.rand(samples) * (range_y[1] - range_y[0]) + range_y[0]

    if weights is None:
        mapping_order = np.random.randint(0, map_count, steps * samples)
    else:
        mapping_order = np.random.choice(
            np.arange(map_count), steps * samples, p=weights)
    mapping_order = mapping_order.reshape((samples, steps))

    for i in range(samples):
        for j in range(steps):
            k = mapping_order[i, j]
            x[i], y[i] = (transformations[k] @ (x[i], y[i])
                          + (shifts[k][0], shifts[k][1]))

    return x, y
