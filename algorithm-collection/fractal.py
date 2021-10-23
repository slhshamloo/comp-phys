import numpy as np

from typing import Optional, Sequence, Union
from math import pi, sin, cos, atan

from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap


def fractal_shape(
        init_x: Sequence[float], init_y: Sequence[float],
        scales: Sequence[Union[np.ndarray, tuple[float, float]]],
        rotations: Sequence[float],
        shifts: Sequence[Union[np.ndarray, tuple[float, float]]],
        steps: int=6) -> tuple[np.ndarray, np.ndarray]:
    """Generates the points of a fractal by repeatedly applying mappings

    Args:
        init_x: initial x coordinates
        init_y: initial y coordinates
        scales: the scalings of the shape in each mappping of the fractal
        rotations: the angles of rotation in each mapping of the fractal
        shifts: the shifts in coordinates of the shape in each mapping
            of the fractal
        steps: the number of times the fractal mapping is applied

    Returns:
        two numpy arrays containing the x and y coordinates of the fractal
        in correct order for drawing the constituent polygons/lines
    """
    transformations = [
        [[scales[i][0] * cos(rotations[i]), -scales[i][0] * sin(rotations[i])],
         [scales[i][1] * sin(rotations[i]), scales[i][1] * cos(rotations[i])]]
        for i in range(len(scales))]
    return fractal_transform(init_x, init_y, transformations, shifts, steps)


def fractal_transform(
        init_x: Sequence[float], init_y: Sequence[float],
        transformations: Sequence[Union[np.ndarray, tuple[
            tuple[float, float], tuple[float, float]]]],
        shifts: Sequence[Union[np.ndarray, tuple[float, float]]],
        steps: int=6) -> tuple[np.ndarray, np.ndarray]:
    """Generates the points of a fractal by repeatedly applying mappings

    The mappings consist of a transformation (transform) and a shift.
    (xf, yf) = (transform)(xi, yi) + shift
    transform = ((a, b), (c, d))
    shift = (shiftx, shifty)

    Args:
        init_x: initial x coordinates
        init_y: initial y coordinates
        transformations: transformations applied to points before shifting
            in each mapping
        shifts: the shifts in coordinates of the shape in each mapping
            of the fractal
        steps: the number of times the fractal mapping is applied

    Returns:
        two numpy arrays containing the x and y coordinates of the fractal
        in correct order for drawing the constituent polygons/lines
    """
    points = np.array([init_x, init_y])

    for _ in range(steps):
        points = np.concatenate(
            [transformations[i] @ points + [[shifts[i][0]], [shifts[i][1]]]
             for i in range(len(transformations))], axis=1)

    return points[0], points[1]


def fractal_point(init_x: Sequence[float], init_y: Sequence[float],
                  scales: Sequence[tuple[float, float]],
                  rotations: Sequence[float],
                  shifts: Sequence[tuple[float, float]],
                  steps: int = 6) -> tuple[list[float], list[float]]:
    """Generates the points of a fractal

    Args:
        init_x: initial x coordinates
        init_y: initial y coordinates
        scales: the scalings of the coordinates in each mappping of the fractal
        rotations: the angles of rotation in each mapping of the fractal
        shifts: the shifts in coordinates for each mapping of the fractal
            as a multiple of the length of the lines in each step
        steps: the number of times the fractal mapping is applied

    Returns:
        two lists containing the x and y coordinates of the fractal
    """
    x = list(init_x)
    y = list(init_y)
    for _ in range(steps):
        i = 0
        while i < len(x) - 1:
            _fractal_iter(x, y, i, scales, rotations, shifts)
            # skip the added points and increment the index by one
            i += len(scales) + 1
    return x, y


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


def fractal_random_scalerot(
        range_x: Sequence[Union[np.ndarray, tuple[float, float]]],
        range_y: Sequence[Union[np.ndarray, tuple[float, float]]],
        scales: Sequence[Union[np.ndarray, tuple[float, float]]],
        rotations: Sequence[float],
        shifts: Sequence[Union[np.ndarray, tuple[float, float]]],
        steps: int = 6, samples=10000,
        weights: Optional[Sequence[float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Generates the points of a fractal by randomly applying mappings

    Applies fractal mappings randomly to randomly sampled points in the
    given range to approximately make the fractal.

    Args:
        range_x: range of the sampling points in the x axis
        range_y: range of the sampling points in the y axis
        scales: the scalings of the shape in each mappping of the fractal
        rotations: the angles of rotation in each mapping of the fractal
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
    transformations = [
        ((scales[i][0] * cos(rotations[i]), -scales[i][0] * sin(rotations[i])),
         (scales[i][1] * sin(rotations[i]), scales[i][1] * cos(rotations[i])))
        for i in range(len(scales))]
    return fractal_random(
        range_x, range_y, transformations, shifts, steps, samples, weights)


def get_polygons(x: Sequence[float], y: Sequence[float], sides: int=3,
                 color: Union[Colormap, str, Sequence[str],
                              tuple[float, float, float, float],
                              Sequence[tuple[float, float, float, float]]]='b'
                 ) -> PolyCollection:
    """
    Generates PolyCollection of polygons from the coordinates of their points

    Args:
        x: x coordinates of polygons' points (in order)
        y: y coordinates of polygons' points (in order)
        sides: the number of sides of the polygons
        color: Color of the polygons. Can be one color, a list of colors,
            or a colormap; If colormap is given, the polygons are colored
            according to their index, using equally spaced values from
            the colormap.

    Returns:
        PolyCollection of the polygons
    """
    if isinstance(color, Colormap):
        colors = [color(i * sides / len(x)) for i in range(len(x) // sides)]
    else:
        colors = color

    xy = np.array([x, y])
    xy = xy.transpose()
    xy = np.split(xy, np.arange(0, len(x), sides))

    return PolyCollection(xy, facecolors=colors, lw=0)


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


def sierpinski_random(range_x: tuple[float, float],
                      range_y: tuple[float, float],
                      steps: int = 10, samples: int=10000
                      ) -> tuple[np.ndarray, np.ndarray]:
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


def barnsley_fern(range_x: Sequence[float], range_y: Sequence[float],
                  steps: int = 20, samples: int = 20000
                  ) -> tuple[np.ndarray, np.ndarray]:
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


def heighway_dragon(init_x: Sequence[float], init_y: Sequence[float],
                    steps: int = 6) -> tuple[list[float], list[float]]:
    """Generates the points of a heighway dragon fractal

    Args:
        init_x: initial x coordinates
        init_y: initial y coordinates
        steps: the number of times the fractal mapping is applied

    Returns:
        two lists containing the x and y coordinates of the fractal
    """
    x = list(init_x)
    y = list(init_y)

    scale = (2 ** (-0.5),) * 2
    shift = (0, 0)
    for _ in range(steps):
        i = 0
        rotation = -pi / 4
        while i < len(x) - 1:
            _fractal_iter_single(x, y, i, scale, rotation, shift)
            # skip the added point and increment the index by one
            i += 2
            rotation *= -1
    return x, y


def _fractal_iter(x: list, y: list, index: int,
                  scales: Sequence[tuple[float, float]],
                  rotations: Sequence[float],
                  shifts: Sequence[tuple[float, float]]) -> None:
    """Applies one iteration of the fractal mapping for one point

    Args:
        x: x coordinates of the fractal's points
        y: y coordinates of the fractal's points
        index: the index of the point's coordinates (in x and y lists)
        scales: the scalings of the coordinates in each mappping of the fractal
        rotations: the angles of rotation in each mapping of the fractal
        shifts: the shifts in the coordinates for each mapping of the
            fractal as a multiple of the length of the lines in each step
    """
    len_x = [(x[index + 1] - x[index]) * scale[0] for scale in scales]
    len_y = [(y[index + 1] - y[index]) * scale[1] for scale in scales]
    length = [(len_x[i] ** 2 + len_y[i] ** 2) ** 0.5
              for i in range(len(len_x))]

    for i in range(len(len_x)):
        rot_angle = _vect_angle(len_x[i], len_y[i]) + rotations[i]

        x.insert(index + 1 + i,
                 x[index] + length[i] * cos(rot_angle)
                 + len_x[i] * shifts[i][0])
        y.insert(index + 1 + i, 
                 y[index] + length[i] * sin(rot_angle)
                 + len_y[i] * shifts[i][1])


def _fractal_iter_single(x: list, y: list, index: int,
                         scale: tuple[float, float],
                         rotation: Sequence[float],
                         shift: tuple[float, float]) -> None:
    """Applies one iteration of fractal mapping for one point

    Args:
        x: x coordinates of the fractal's points
        y: y coordinates of the fractal's points
        index: the index of the point's coordinates (in x and y lists)
        scale: scaling of the coordinates
        rotation: angle of rotation
        shift: shift in coordinates as a multiple of the length of the lines
    """
    len_x = (x[index + 1] - x[index]) * scale[0]
    len_y = (y[index + 1] - y[index]) * scale[1]
    length = (len_x ** 2 + len_y ** 2) ** 0.5
    rot_angle = _vect_angle(len_x, len_y) + rotation

    x.insert(index + 1,
             x[index] + length * cos(rot_angle)
             + len_x * shift[0])
    y.insert(index + 1, 
             y[index] + length * sin(rot_angle)
             + len_y * shift[1])


def _vect_angle(x: float, y: float) -> float:
    """Calculates the angle of the vector (len_x, len_y) with +x

    Args:
        len_x: x component of the vector
        len_y: y component of the vector
    
    Returns:
        the angle of (len_x, len_y) with the positive direction of the x axis 
    """
    if x == 0:
            if y > 0:
                return pi / 2
            else:
                return -pi / 2
    else:
        angle = atan(y / x)
        if x < 0:
            angle += pi
        
        return angle
