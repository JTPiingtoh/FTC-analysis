import numpy as np
from PIL import Image
from roifile import ImagejRoi

from shapely import Polygon, box, intersection

import math



class LISEE_ZONE():
    """FTC measurement zones"""
    def MEDIAL(roi_width: float|int): 
        return - 0.5 * (1 / 3.0) * roi_width
    
    def LATERAL(roi_width: float|int): 
        return 0.5 * (1 / 3.0) * roi_width


    
# print(HORI_ZONE.MEDIAL)

# returns polygon representing roi with 12.5% of roi removed from each end
def trim_roi_polygon(
        roi: ImagejRoi,
        image_height: int|float,
        trim_factor: float
) -> Polygon:
    '''
    Returns coordinates of a trimmed roi
    '''
    roi_width = roi.left - roi.right
    new_roi_left = roi.right - (roi_width * trim_factor) # Left and right are seemingly flipped
    new_roi_right = roi.right + (roi_width * trim_factor)
    trim_box = box(new_roi_left, 0, new_roi_right, image_height)

    # print(new_roi_left)

    return intersection(trim_box, Polygon(roi.integer_coordinates)) # Intersect betweeb trim box and roi


def box_polygon_intersect(
        box_coords: tuple[float, float, float, float],
        polygon: Polygon
) -> Polygon:
    '''
    Convenience helper function for lisee_CSA_polygon_function. Takes a Polygon and box coordinates,
    producing a polygon representing their overlap
    '''
    xmin, ymin, xmax, ymax = box_coords

    return intersection(box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), polygon)


# TODO: move to own file
def lisee_CSA_polygon_function(
        image_height: float|int,
        image_width: float|int,
        roi_coords_x: np.ndarray,
        roi_coords_y: np.ndarray,
        roi_mid_x: float|int         
) -> tuple[Polygon, Polygon, Polygon]:
    
    roi_left_x = float(min(roi_coords_x))
    roi_right_x = float(max(roi_coords_x))
    roi_width = float(roi_left_x - roi_right_x)

    medial_adjust: float = LISEE_ZONE.MEDIAL(roi_width)
    lateral_adjust: float = LISEE_ZONE.LATERAL(roi_width)

    medial_x = float(roi_mid_x + medial_adjust)
    lateral_x = float(roi_mid_x + lateral_adjust)

    trimmed_roi_polygon = Polygon(np.column_stack((roi_coords_x, roi_coords_y)))

    medial_roi_polygon = box_polygon_intersect(
        box_coords=(roi_left_x, 0, lateral_x, image_height),
        polygon=trimmed_roi_polygon)

    central_roi_polygon = box_polygon_intersect(
        box_coords=(lateral_x, 0, medial_x, image_height),
        polygon=trimmed_roi_polygon
    )

    lateral_roi_polygon = box_polygon_intersect(
        box_coords=(medial_x, 0, roi_right_x, image_height),
        polygon=trimmed_roi_polygon
    )
    
    assert isinstance(central_roi_polygon, Polygon)
    assert isinstance(medial_roi_polygon, Polygon)
    assert isinstance(lateral_roi_polygon, Polygon)

    return lateral_roi_polygon, central_roi_polygon, medial_roi_polygon


def line_length(
    coords_x: np.ndarray,
    coords_y: np.ndarray
)-> float:
    '''
    Takes an ordered series of coordinates, and returns the total length of lines
    drawn between them.
    '''
    if coords_x.size != coords_y.size:
        raise RuntimeError("line_length recieved coordinate arrays if different sizes.")

    if coords_x.ndim != 1 and coords_y.ndim != 1:
        raise RuntimeError(f"line_length takes coord arrays with 1 dimension. coords_x.ndim: {coords_x.ndim}, coords_y.ndim:{coords_y.ndim}")

    total_length:float = 0.0

    for i in range(coords_x.shape[0] - 1):
        x1, y1 = coords_x[i], coords_y[i]
        x2, y2 = coords_x[i + 1], coords_y[i + 1]

        # print(f"{int(x1), int(y1)}, {int(x2), int(y2)}")

        total_length += math.sqrt(
            math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)
        )

    return total_length


# Return lisee csa measures
def lisee_zone_average_thickness(
    lisee_zone: Polygon  
        ) -> float:
    '''
    Takes a lisee zone, and returns its average thickness as determined by the bottom line, or the bone
    -cartilage interface.
    '''
    zone_coords_x, zone_coords_y = np.array(lisee_zone.exterior.coords.xy)

    assert zone_coords_x.size > 0 and zone_coords_y.size > 0
    assert zone_coords_x.shape == zone_coords_y.shape
    assert zone_coords_x.ndim == 1 and zone_coords_y.ndim == 1

    left_x = min(zone_coords_x)
    right_x = max(zone_coords_x)

    left_indexes = []
    right_indexes = []

    for i in range(zone_coords_x.shape[0]):
        if zone_coords_x[i] == left_x:
            left_indexes.append(i)
        elif zone_coords_x[i] == right_x:
            right_indexes.append(i)
    
    if len(left_indexes) == 1:
        raise RuntimeError("lisee_zone_average_thickness polygon does not have a flat lhs")

    if len(right_indexes) == 1:
        raise RuntimeError("lisee_zone_average_thickness polygon does not have a flat rhs")

    left_index = left_indexes[0]
    max_y_left_indexes = zone_coords_y[left_index]
    for index in left_indexes:
        if zone_coords_y[index] > max_y_left_indexes:
            max_y_left_indexes = zone_coords_y[index]
            left_index = index
    
    right_index = right_indexes[0]
    max_y_right_indexes = zone_coords_y[right_index]
    for index in right_indexes:
        if zone_coords_y[index] > max_y_right_indexes:
            max_y_right_indexes = zone_coords_y[index]
            right_index = index

    index_1 = left_index
    index_2 = right_index

    if index_1 == index_2:
        raise RuntimeError("Lisee_zone_average_thickness failed to find bottom line ends.")
    
    # ensures indexes are ordered correctly when slicing coord arrays. 
    if index_1 > index_2:
        index_2, index_1 = index_1, index_2

    return lisee_zone.area / line_length(coords_x=zone_coords_x[index_1:index_2 + 1], coords_y=zone_coords_y[index_1:index_2 + 1])