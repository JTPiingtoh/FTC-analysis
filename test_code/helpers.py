import numpy as np


from shapely import  intersection, LineString


'''
Finds the roi midpoint using an algorithm
'''

def intersecting_segment_coords(      
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    line: LineString,
    intersect_loc: str = "top",
    return_intersect_coord : bool = False
    # image_height: float|int
) -> int | tuple[int, tuple[float, float]]:
    '''From a list of coords, returns index of the coordinate with the highest or lowest y value of ROI points that intersect 
    a (vertical) line. Used to find points that intersect with the ROI top or bottom line, hori zones etc'''

    assert roi_coords_x.shape == roi_coords_y.shape
    assert intersect_loc in ["top", "bottom"]

    max_y = roi_coords_y.max()
    min_y = roi_coords_y.min()
    intersect_coord = 0
    point_index = 0
    

    for i in range(roi_coords_x.shape[0] - 1):

        point1 = roi_coords_x[i], roi_coords_y[i]
        point2 = roi_coords_x[i+1], roi_coords_y[i+1]

        poly_segment = LineString([point1, point2])

        intersect = intersection(line, poly_segment)

        if intersect:

            # print(f"intersect_xy: {intersect.coords.xy}")
            
            intersect_y = intersect.coords.xy[1][0]

            # print(f"intersect_y: {intersect_y}")
            
            if intersect_loc == "top":
                if not intersect_y < max_y:
                    continue
            
            elif intersect_loc == "bottom":
                if not intersect_y > min_y:
                    continue

            point_index = i
            max_y = intersect_y
            min_y = intersect_y
            intersect_coord = (intersect.coords.xy[0][0], intersect.coords.xy[1][0])
            

    if return_intersect_coord:
        return point_index, intersect_coord

    return point_index


def line_start_index(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    intersect_loc = "top"
):
    '''
    Finds a relatively central point of the ROI topline in the form of an index. Made for use with swingloops.
    '''

    assert roi_coords_x.shape == roi_coords_y.shape
    assert intersect_loc in ["top", "bottom"]

    roi_left_x = float(min(roi_coords_x))
    roi_right_x = float(max(roi_coords_x))
    roi_top_y = float(min(roi_coords_y))
    roi_bottom_y = float(max(roi_coords_y))
    roi_width = float(roi_right_x - roi_left_x)


    # Find topline using vertical ray from roi midpoint #         
    roi_halfway_vertical_line = LineString(
        [[roi_left_x + (roi_width/2.0), roi_bottom_y ], 
        [roi_left_x + (roi_width/2.0), roi_top_y]]
    )

    line_intersect_index = intersecting_segment_coords(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        line=roi_halfway_vertical_line,
        intersect_loc=intersect_loc
    )

    return line_intersect_index

