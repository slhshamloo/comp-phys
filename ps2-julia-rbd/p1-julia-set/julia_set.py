from typing import Callable
import numpy as np


def julia_set(function: Callable, thresh: float,
              bounds: tuple[complex, complex], iter: int = 256,
              res_re: int = 2048, res_im: int = 2048) -> np.ndarray:
    """Generates the Julia set for the given function

    source: https://codereview.stackexchange.com/a/210445

    Parameters
    ----------
    function : Callable
        the complex mapping function
    thresh : float
        The threshold for the absolute value of the numbers;
        If the absolute value of a number exceeds this, it is considered
        that the point has diverged.
    bounds : tuple[complex, complex]
        The start and end points of the domain; The real part of the
        numbers is between range[0].real and range[1].real (inclusive) and
        the imaginary part of the numbers is between range[0].imag and
        range[1].imag (inclusive)
    iter : int, optional
        the number of times the function is applied, by default 10
    res_re : int, optional
        the number of elements in the real axis, by default 1080
    res_im : int, optional
        the number of elements in the imaginary axis, by default 1080

    Returns
    -------
    numpy.ndarray
        res by res 2D array where each element is the number of iteration
        applied before each point diverges, and the elements representing
        the points that don't diverge with this many iterations are equal
        to iter (the number of iterations)
    """
    im, re = np.ogrid[-bounds[0].imag:bounds[1].imag:complex(0, res_im),
                      -bounds[0].real:bounds[1].real:complex(0, res_re)]
    z = (re + 1j * im).flatten()

    converging = np.indices(z.shape)
    julia = np.empty_like(z, dtype=int)

    for i in range(1, iter + 1):
        z[converging] = function(z[converging])
        diverging = abs(z[converging]) > thresh
        julia[converging[diverging]] = i
        converging = converging[~diverging]

    julia[converging] = iter
    return julia.reshape(res_im, res_re)


def julia_quad(
    c: complex, iter: int = 256, res: int = 2048) -> np.ndarray:
    """
    Generates the Julia set for the complex quadratic polynomial f(z) = z^2 + c

    The domain is the square with vertices -R, R, -iR, iR, where
    R = (1 + sqrt(1 + 4|c|)) / 2, since the points outside this area will
    definitely diverge (in fact, points outside the circle centered at
    the origin with radius R will definitely diverge, but since we work
    with rectangular displays and arrays, we choose the square domain).

    Parameters
    ----------
    c : complex
        the constant added to z^2 in the complex quadratic polynomial
    iter : int, optional
        the number of times the function is applied, by default 10
    res : int, optional
        The resolution of the output; The output's size is res x res;
        By default 1080.

    Returns
    -------
    numpy.ndarray
        res by res 2D array where each element is the number of iteration
        applied before each point diverges, and the elements representing
        the points that don't diverge with this many iterations are equal
        to iter (the number of iterations)
    """
    radius = (1 + (1 + 4 * abs(c)) ** 0.5) / 2
    return julia_set(lambda z : z**2 + c, radius, (radius + 1j * radius,) * 2,
                     iter, res, res)


def crop_julia(julia_set: np.ndarray) -> np.ndarray:
    """Crops the given julia set to exclude the unnecessary points

    Parameters
    ----------
    julia_set : np.ndarray
        The julia set array, where each element is equal to the number of
        iterations befor diverging; See the julia_set function output.

    Returns
    -------
    numpy.ndarray
        The cropped julia set
    """
    mask = np.argwhere(julia_set > 1)
    re_min, im_min = mask.min(0)
    re_max, im_max = mask.max(0)
    return julia_set[re_min:re_max, im_min:im_max]
