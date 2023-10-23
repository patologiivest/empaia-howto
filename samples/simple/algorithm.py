import numpy as np


def algorithm(tiles, roi):
    result = 0
    for t in tiles:
        a = np.array((t))  # convert image to width x height x 3 array of pixel color values
        result += np.cumsum(a) / np.cumprod(a)  # silly calculation for demontration purposes

    return 42.0 + result  # whatever ...
