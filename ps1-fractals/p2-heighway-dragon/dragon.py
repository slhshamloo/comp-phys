from math import pi, sin, cos, atan
from typing import Sequence


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
            _fractal_iter(x, y, i, scale, rotation, shift)
            # skip the added point and increment the index by one
            i += 2
            rotation *= -1
    return x, y


def _fractal_iter(x: list, y: list, index: int,
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
