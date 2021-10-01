import numpy as np


def random_ballistic_deposition(surface: np.ndarray, time: int, layers: int
                                ) -> np.ndarray:
    """
    Randomly increments the surface array elements and returns layer heights

    Parameters
    ----------
    surface : numpy.ndarray
        array of the number of particles deposited on each point on
        the surface
    time : int
        the amount of time passed, which is equal to the number of
        added particles (increments)
    layers : int
        the number of layers

    Returns
    -------
    numpy.ndarray
        array of heights for every layer (array of surface arrays)
    """
    point_sample = np.random.randint(len(surface.flat), size=time)
    layer_heights = np.empty((layers,) + surface.shape, dtype=int)

    for i in range(layers):
        for j in range(time // layers):
            surface.flat[point_sample[i * time // layers + j]] += 1
        layer_heights[i] = surface

    # the last layer, if the particles can't be divided evenly between layers
    if time - time // layers * layers > 0:
        for i in range(time - time // layers * layers):
            surface.flat[point_sample[time // layers * layers + i]] += 1
        layer_heights[-1] = surface

    return layer_heights
