from matplotlib.colors import LinearSegmentedColormap


def wbrykmap(anchors: tuple[float, float, float] = (1/4, 1/2, 3/4)
             ) -> LinearSegmentedColormap:
    """
    Custom colormap that goes from white to blue to red to yellow to black

    inspired by http://paulbourke.net/fractals/juliaset/

    Parameters
    ----------
    anchors : tuple[float, float, float], optional
        anchor points of the blue, purple, and red colors respectively,
        by default (1/4, 1/2, 3/4)

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Colormap usable by matplotlib
    """
    cdict = {
        "red": [[0, 1, 1], [anchors[0], 0, 0], [anchors[1], 1, 1],
                [anchors[2], 1, 1], [1, 0, 0]],
        "blue": [[0, 1, 1], [anchors[0], 1, 1], [anchors[1], 0, 0],
                 [anchors[2], 0.5, 0.5], [1, 0, 0]],
        "green": [[0, 1, 1], [anchors[0], 0.5, 0.5], [anchors[1], 0, 0],
                  [anchors[2], 1, 1], [1, 0, 0]],
    }
    return LinearSegmentedColormap("wrbpk", cdict)
