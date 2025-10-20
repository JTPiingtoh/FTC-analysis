import numpy as np
from shapely import Polygon, box, intersection



def trim_roi_coords(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    image_width: int|float,
    image_height: int|float,
    trim_factor: float,
    image_left_coord: int|float
) -> np.ndarray:
    '''
    Returns ndarray representing coords of trimmed roi based on image width. Ensure roi supplied to function has coords aligned with background image. 
    '''

    new_roi_left: int = image_width * (trim_factor/ 2) # Left and right are seemingly flipped
    # print(new_roi_left)

    new_roi_right: int = image_width - image_width * (trim_factor/ 2)

    trim_box = box(new_roi_left+image_left_coord, 0, new_roi_right, image_height)

    intersection_polgyon= intersection(trim_box, Polygon(np.column_stack((roi_coords_x, roi_coords_y))), grid_size=None)

    return np.array(intersection_polgyon.exterior.coords.xy)


def trim_roi_coords_roi_based(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    roi_left: int|float,
    roi_right: int|float,
    trim_factor: float,
    image_height: int|float
) -> np.ndarray:
    '''
    Returns ndarray representing coords of trimmed ROI based on ROI width. Ensure ROI supplied to function has coords aligned with background image. 
    '''

    roi_width = roi_right - roi_left

    new_roi_left: int = roi_left + roi_width * (trim_factor/ 2) # Left and right are seemingly flipped
    # print(new_roi_left)

    new_roi_right: int = roi_right - roi_width * (trim_factor/ 2)

    trim_box = box(new_roi_left, 0, new_roi_right, image_height)

    intersection_polgyon = intersection(trim_box, Polygon(np.column_stack((roi_coords_x, roi_coords_y))), grid_size=None)

    return np.array(intersection_polgyon.exterior.coords.xy)