from shapely import Polygon, contains_xy
import numpy as np

def average_ei(
        image_width: int,
        image_height: int,
        polygon: Polygon,
        image_array: np.ndarray
) -> float:
    '''
    Calculates the average echo intensity of an image region defined by a shapely polygon.
    '''

    assert image_array.ndim == 2 # rgb channels should be converted to single greyscale dim

    average_ei: float = 0
    n: int = 0

    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):

            if not contains_xy(polygon, y, x):
                continue

            echo_intensity = image_array[x][y]
            if 0 > echo_intensity > 255:
                raise RuntimeError("Supplied array has invalid echo intensities")
            
            average_ei = ((average_ei * n) + echo_intensity) / (n + 1)
            n = n + 1

    return average_ei