from math import pi, sin, cos, atan
from typing import Sequence


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
