from math import pi, sin, cos, atan
from typing import Sequence, Union

from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap

import numpy as np


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
