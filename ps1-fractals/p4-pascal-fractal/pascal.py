import numpy as np
from typing import Union
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colors import Colormap, LogNorm


def pascal_triangle(size: int) -> np.ndarray:
    """Creates an array representation of pascal's triangle (right-angled)

    Args:
        size: width and height of the triangle

    Returns:
        A 2D numpy array containing the triangle. The triangle is
        right-angled and the elements where the triangle is not defined
        are filled with zeros.
        
        Example: Triangle of size 5
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 2, 1, 0, 0],
         [1, 3, 3, 1, 0],
         [1, 4, 6, 4, 1]]
    """
    grid = np.zeros((size, size), dtype=np.uint64)
    grid[:, 0] = 1
    for i in range(1, size):
        for j in range(1, size):
            grid[i, j] = grid[i - 1, j - 1] + grid[i - 1, j]
    return grid


def draw_pascal_triangle(ax: Axes, size: int, fontsize: int = 10,
                         normalize: bool = True) -> None:
    """Plots pascal's triangle with the given size on ax

    Args:
        ax: matplotlib axes object
        size: width and height of the triangle
        fontsize: font size of the numbers
        normalize: whether or not to normalize the coordinates
            and the axes' limits to make the triangle equilateral
    """
    grid = pascal_triangle(size)
    hor = 1
    if normalize:
        ver = 3**0.5 / 2 * hor
    else:
        ver = hor

    for i in range(size):
        for j in range(i + 1):
            ax.text((size - i + 2 * j) * hor,
                    (2 * (size - 1 - i) + 1) * ver,
                    str(int(grid[i][j])), fontsize=fontsize,
                    ha="center", va="center")

    if normalize:
        ax.set_xlim(0, 2 * hor * (size - 1))
        ax.set_ylim(0, 2 * ver * (size - 1))


def draw_pascal_fractal(ax: Axes, size: int,
                        draw_numbers: bool = True, fontsize: float = 10,
                        cmap: Union[Colormap, str]="rainbow") -> AxesImage:
    """Draws the serpinski triangle inside pascal's triangle

    This is achieved by connecting the odd numbers in pascal's triangle

    Args:
        ax: matplotlib axes object
        size: width and height of the triangle
        draw_numbers: wheter or not to draw pascal's triangle
            on top of the fractal
        fontsize: font size of the numbers
        cmap: The colormap used for coloring the points. The points are
            colored according to the ratio of their number to the maximum
            number in the triangle. Logarithmic normalization is applied
            for better distinction.

    Returns:
        output of imshow
    """
    grid = pascal_fractal(size)

    image = ax.imshow(np.flip(grid, axis=0), origin="lower",
                      extent=(0, 2 * size, 0, 2 * size),
                      norm=LogNorm(), cmap=cmap)
    if draw_numbers:
        draw_pascal_triangle(ax, size, fontsize, False)

    ax.set_aspect('auto')
    return image


def pascal_fractal(size: int) -> np.ndarray:
    """Masked array representation of pascal's triangle serpinski fractal

    Suitable for use with imshow

    Args:
        size: width and height of the triangle

    Returns:
        Masked numpy array containing the serpinski triangle inside
        pascal's triangle. The numbers are repeated four times in squares
        and the squares are centered. Even numbers are masked.

        Example: Triangle of size 5
        [[- - - - 1 1 - - - -]
         [- - - - 1 1 - - - -]
         [- - - 1 1 1 1 - - -]  
         [- - - 1 1 1 1 - - -]  
         [- - 1 1 - - 1 1 - -]  
         [- - 1 1 - - 1 1 - -]
         [- 1 1 3 3 3 3 1 1 -]
         [- 1 1 3 3 3 3 1 1 -]
         [1 1 - - - - - - 1 1]
         [1 1 - - - - - - 1 1]]
    """
    grid = _pascal_triangle_map(size)
    return np.ma.masked_where(grid % 2 == 0, grid)


def _pascal_triangle_map(size: int) -> np.ndarray:
    """Array representation of pascal's triangle, centered

    Args:
        size (int): width and height of the triangle

    Returns:
        Numpy array containing centered pascal's triangle. The numbers
        are repeated four times in squares and the squares are centered.
        Elements where the triangle is not defined are filled with zeros.

        Example: Triangle of size 5
        [[0 0 0 0 1 1 0 0 0 0]
         [0 0 0 0 1 1 0 0 0 0]
         [0 0 0 1 1 1 1 0 0 0]
         [0 0 0 1 1 1 1 0 0 0]
         [0 0 1 1 2 2 1 1 0 0]
         [0 0 1 1 2 2 1 1 0 0]
         [0 1 1 3 3 3 3 1 1 0]
         [0 1 1 3 3 3 3 1 1 0]
         [1 1 4 4 6 6 4 4 1 1]
         [1 1 4 4 6 6 4 4 1 1]]
    """
    grid = pascal_triangle(size)
    grid = grid.repeat(2, axis=0)
    grid = grid.repeat(2, axis=1)

    shifts = np.linspace(size - 1, 0, size).astype(int)
    shifts = shifts.repeat(2)
    return _apply_shifts(grid, shifts)


def _apply_shifts(array: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Applies different shifts to diffrent rows of a numpy array

    Args:
        array: the numpy array
        shifts: shift amounts as a numpy array

    Returns:
        the numpy array with shifted rows
    """
    row, col = np.ogrid[:array.shape[0], :array.shape[1]]
    col = col - shifts[:, np.newaxis]
    return array[row, col]
