import numpy as np
from numpy.polynomial import polynomial
from PIL import Image

from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point, contains_xy, LineString
from shapely.plotting import plot_line, plot_polygon, plot_points
import matplotlib.pyplot as plt

from midpoint_from_algo import roi_midpoint_from_algo
from roi_rotate import roi_leftmost_rightmost, rotate_image_and_roi
from roi_trim import trim_roi_coords_roi_based
from conversions import mm_to_pixels, pixels_to_mm
from swingloop import CycleLoop
from test_code.helpers import intersecting_segment_coords, line_start_index

import sympy

# Defines hori zones as per mm to pixel conversion
# TODO: Is mixing a function with a class good design?
class HORI_ZONE():
    
    MEDIAL: float = - mm_to_pixels(10)
    LATERAL: float = mm_to_pixels(10)


def get_closest_point2d(p1, p2, p3):

    '''
    Where p1 is a single point, and p2 and p3 are points of a line segment

    Adapted from: 
    https://math.stackexchange.com/questions/2483734/get-point-from-line-segment-that-is-closest-to-another-point, 
    https://www.desmos.com/calculator/qbtyssnnmf and 
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    '''

    p1x, p1y = p1
    p2x, p2y = p2
    p3x, p3y = p3

    A = p1x - p2x
    B = p1y - p2y
    C = p3x - p2x
    D = p3y - p2y

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1

    if (len_sq != 0): # in case of 0 length line
        param = dot / len_sq

    ax = p3x - p1x
    ay = p3y - p1y

    bx = p2x - p3x
    by = p2y - p3y

    t = - ( (ax * bx) + (ay * by) ) / ( bx ** 2 + by ** 2)

    if 0 < param <= 1:
        p4x = p3x + t * (p2x - p3x)
        p4y = p3y + t * (p2y - p3y)
        return (p4x, p4y)
    elif param < 0:
        return (p2x, p2y)
    else:
        return (p3x, p3y)
    


# BUG: distance needs to be bounded
# TODO: prevent iterating over same points
def get_hori_coords_thickness(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    roi_mid_x: float|int,
    return_thickness_px: bool = True
):
    '''
    Used for obtaining the LHS and RHS coords for the hori ROI. Also returns the medial and lateral thickness measurements.
    '''

    hori_lateral_x = roi_mid_x + HORI_ZONE.LATERAL
    hori_medial_x = roi_mid_x + HORI_ZONE.MEDIAL
    roi_top_x = float(min(roi_coords_y))
    roi_bottom_x = float(max(roi_coords_y))

    _, hori_lateral_bottom_coord = intersecting_segment_coords(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        line=LineString([[hori_lateral_x, roi_bottom_x ], 
        [hori_lateral_x, roi_top_x]]),
        intersect_loc="bottom",
        return_intersect_coord=True
    )

    _, hori_medial_bottom_coord = intersecting_segment_coords(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        line=LineString([[hori_medial_x, roi_bottom_x ], 
        [hori_medial_x, roi_top_x]]),
        intersect_loc="bottom",
        return_intersect_coord=True
    )

    start_index = line_start_index(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        intersect_loc="top"
    )

    top_roi_coords_x_cycle = CycleLoop(iterable=roi_coords_x, start_index=start_index)
    top_roi_coords_y_cycle = CycleLoop(iterable=roi_coords_y, start_index=start_index)
    
    iterator = 1
    edges = 0
    iterations = 0
    MAX_ITERATIONS = 200

    hori_lateral_max_weighted_distance = 0 # use inverse of distance to allow comparion with 0
    hori_lateral_thickness_px = 0
    hori_lateral_top_coord = 0

    hori_medial_max_weighted_distance = 0
    hori_medial_thickness_px = 0
    hori_medial_top_coord = 0

    while edges < 2:

        if iterations > MAX_ITERATIONS:
            raise RuntimeError(f"get_roi_polygon() could not find midpoint after {MAX_ITERATIONS} iterations." \
                               "Roi may not have been trimmed.")
        
        current_x, current_y = top_roi_coords_x_cycle.current, top_roi_coords_y_cycle.current
        next_x, next_y = top_roi_coords_x_cycle.peek1(iterator), top_roi_coords_y_cycle.peek1(iterator) 

        # detects if edge reached
        if next_x == current_x:
            iterator *=- 1
            edges += 1
            top_roi_coords_x_cycle.set_curret(start_index + iterator)
            top_roi_coords_y_cycle.set_curret(start_index + iterator)
            continue

        p2 = sympy.Point2D(current_x, current_y, evaluate=False)
        p3 = sympy.Point2D(next_x, next_y, evaluate=False)

        medial_p4 = get_closest_point2d(hori_medial_bottom_coord, p2 , p3)
        medial_distance = np.linalg.norm(np.array(medial_p4, dtype = 'float') - np.array(hori_medial_bottom_coord, dtype = 'float'))

        lateral_p4 = get_closest_point2d(hori_lateral_bottom_coord, p2 , p3)
        lateral_distance = np.linalg.norm(np.array(lateral_p4, dtype = 'float') - np.array(hori_lateral_bottom_coord, dtype = 'float'))

        lateral_weight = 1.0 / lateral_distance
        medial_weight = 1.0 / medial_distance

        if medial_weight > hori_medial_max_weighted_distance:

            hori_medial_max_weighted_distance = medial_weight

            hori_medial_thickness_px = medial_distance
            hori_medial_top_coord = medial_p4
            
        elif lateral_weight > hori_lateral_max_weighted_distance:

            hori_lateral_max_weighted_distance = lateral_weight

            hori_lateral_thickness_px = lateral_distance
            hori_lateral_top_coord = lateral_p4
        
        top_roi_coords_x_cycle.step(iterator)
        top_roi_coords_y_cycle.step(iterator)
        iterations += 1

    hori_coords = (hori_medial_top_coord, hori_medial_bottom_coord, hori_lateral_top_coord, hori_lateral_bottom_coord)
    hori_ML_thicknesses = (hori_medial_thickness_px, hori_lateral_thickness_px)
    
    if return_thickness_px:
        return hori_coords, hori_ML_thicknesses

    return hori_coords


def get_side_polygon(
    image_height,
    image_width,
    top_coord,
    bottom_coord,
    side: str   
):
    
    assert side in ["medial", "lateral"
    ""]
    dx_dy = (
            (top_coord[0] - bottom_coord[0]) 
            / 
            (top_coord[1] - bottom_coord[1]) 
    )

    inveted_c = top_coord[0] - top_coord[1] * dx_dy

    top_intercept_x = dx_dy * image_height + inveted_c
    bottom_intercept_x = inveted_c

    if side == "medial":
        return Polygon(
            [
                [0,0],
                [0, image_height],
                [top_intercept_x, image_height],
                [bottom_intercept_x, 0],
            ]
        )
    
    elif side == "lateral":
        return Polygon(
            [
                [image_width,0],
                [image_width, image_height],
                [top_intercept_x, image_height],
                [bottom_intercept_x, 0],
            ]
        )

def get_hori_csa_px(
    image_height,
    image_width,
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    hori_coords: np.ndarray,
    roi_mid_x: float | int
) -> float:
    '''
    Calculates hori csa of image roi
    '''
    # TODO: Creat polygon made made hori_coords
    hori_medial_top_coord, hori_medial_bottom_coord, hori_lateral_top_coord, hori_lateral_bottom_coord = hori_coords
    
    hori_medial = np.array([hori_medial_top_coord, hori_medial_bottom_coord])
    hori_lateral = np.array([hori_lateral_top_coord, hori_lateral_bottom_coord])

    roi = Polygon(np.vstack([roi_coords_x,roi_coords_y]).T)

    # where lines intercept with the top and bottom of the image
    medial_region = get_side_polygon(
        image_height=image_height,
        image_width=image_width,
        top_coord=hori_medial_top_coord,
        bottom_coord=hori_medial_bottom_coord,
        side="medial"
    )

    lateral_region = get_side_polygon(
        image_height=image_height,
        image_width=image_width,
        top_coord=hori_lateral_top_coord,
        bottom_coord=hori_lateral_bottom_coord,
        side="lateral"
    )

    medial_area_cut = medial_region.intersection(roi).area
    lareal_area_cut = lateral_region.intersection(roi).area

    return roi.area - medial_area_cut - lareal_area_cut


def get_hori_central_thickness_px(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    roi_mid_x: float | int,
    image_height,
) -> float :
    
    roi = Polygon(np.vstack([roi_coords_x,roi_coords_y]).T)
    midLine = LineString(
            [
                [roi_mid_x, 0 ], 
                [roi_mid_x, image_height]
            ]
        )
    
    return roi.intersection(midLine).length
    

if __name__ == "__main__":

    from midpoint_from_algo import roi_midpoint_from_algo

    with TiffFile('14_annotated_with_border.tif') as tif:
        
        image_array = tif.pages[0].asarray()
    
        image_width = image_array.shape[1]
        image_height = image_array.shape[0]
        assert tif.imagej_metadata is not None
        roi_bytes = tif.imagej_metadata['ROI']
        roi = ImagejRoi.frombytes(roi_bytes)
        coords = roi.integer_coordinates
        left = roi.left
        top = roi.top      
        right = roi.right
        bottom = roi.bottom

        coords += [left, top]
        
        # results = FTC_analysis(image,roi)
        # trimmed_roi_polygon = Polygon(np.column_stack(results["trimmed_roi_coords"]))
        rotated_image, rotated_roi_coords = rotate_image_and_roi(image=Image.fromarray(image_array), roi=roi)

        trimmed_roi_coords = trim_roi_coords_roi_based(
        roi_coords_x=rotated_roi_coords[:,0], # every column of the first row
        roi_coords_y=rotated_roi_coords[:,1],
        roi_left=left,
        roi_right=right,
        image_height=image_height,
        trim_factor=0.25
        )

        trimmed_roi_algo_mid_x = roi_midpoint_from_algo(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        )

        fig, ax = plt.subplots()
        # ax.imshow(rotated_image)
        ax.imshow(rotated_image)
        ax.plot(trimmed_roi_coords[0], trimmed_roi_coords[1], 'bo', linestyle='-')

        hori_coords, hori_ML_thicknesses = get_hori_coords_thickness(
            roi_coords_x=trimmed_roi_coords[0], # every column of the first row
            roi_coords_y=trimmed_roi_coords[1],
            roi_mid_x=trimmed_roi_algo_mid_x,
            return_thickness_px=True
        )

        hori_medial_top_coord, hori_medial_bottom_coord, hori_lateral_top_coord, hori_lateral_bottom_coord = hori_coords
        

        hori_csa_px = get_hori_csa_px(
            image_height=image_height,
            image_width=image_width,
            roi_coords_x=trimmed_roi_coords[0],
            roi_coords_y=trimmed_roi_coords[1],
            hori_coords=hori_coords,
            roi_mid_x=trimmed_roi_algo_mid_x,
        )

        hori_central_thickness_px = get_hori_central_thickness_px(
            image_height=image_height,
            roi_coords_x=trimmed_roi_coords[0],
            roi_coords_y=trimmed_roi_coords[1],
            roi_mid_x=trimmed_roi_algo_mid_x,   
        )

        ax.plot([hori_medial_bottom_coord[0], hori_medial_top_coord[0]],[hori_medial_bottom_coord[1], hori_medial_top_coord[1]] , 'ro')
        ax.plot([hori_lateral_bottom_coord[0], hori_lateral_top_coord[0]],[hori_lateral_bottom_coord[1], hori_lateral_top_coord[1]] , 'go')
        print(hori_central_thickness_px)



        plt.show()
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_algo_mid_x"])

        colors = ["red", "green", "blue" ,"purple"]

        # print(f'Central av thickness: {results["lisee_lateral_average_thickness_mm"]}')


