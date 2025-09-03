import numpy as np
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
from helpers import intersecting_segment_coords, line_start_index

from sympy import Point, Line

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

    



def get_hori_polygon(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    roi_mid_x: float|int       
):
    hori_lateral_x = roi_mid_x - HORI_ZONE.LATERAL
    hori_medial_x = roi_mid_x - HORI_ZONE.MEDIAL
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

    # TODO: iterate through top line, find shortest disance

    top_line_intersect_index = line_start_index(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        intersect_loc="top"
    )

    top_roi_coords_x_cycle = CycleLoop(iterable=roi_coords_x, start_index=top_line_intersect_index)
    top_roi_coords_y_cycle = CycleLoop(iterable=roi_coords_y, start_index=top_line_intersect_index)
    
    iterator = 1
    edges = 0
    iterations = 0
    MAX_ITERATIONS = 200

    hori_lateral_closest_top_segment_index = 0
    hori_lateral_max_weighted_distance = 0 # use inverse of distance to allow comparion with 0
    hori_lateral_thickness_px = 0
    hori_top_lateral_point = 0

    hori_medial_closest_top_segment_index = 0
    hori_medial_max_weighted_distance = 0
    hori_medial_thickness_px = 0

    while edges < 2:

        top_roi_coords_x_cycle.step(iterator)
        top_roi_coords_y_cycle.step(iterator)
        iterations += 1

        if iterations > MAX_ITERATIONS:
            raise RuntimeError(f"get_roi_polygon() could not find midpoint after {MAX_ITERATIONS} iterations." \
                               "Roi may not have been trimmed.")
        
        current_x, current_y = top_roi_coords_x_cycle.current, top_roi_coords_y_cycle.current
        next_x, next_y = top_roi_coords_x_cycle.peek1(iterator), top_roi_coords_y_cycle.peek1(iterator) 

        # detects if edge reached
        if next_x == current_x:
            iterator *=- 1
            edges += 1
            continue

        p2 = Point(current_x, current_y)
        p3 = Point(next_x, next_y)

        top_line_segment = Line(p2,p3)

        lateral_distance = top_line_segment.distance(Point(hori_lateral_bottom_coord[0], hori_lateral_bottom_coord[1]))
        medial_distance = top_line_segment.distance(Point(hori_medial_bottom_coord[0], hori_medial_bottom_coord[1]))

        lateral_weight = 1.0 / lateral_distance
        medial_weight = 1.0 / medial_distance

        if lateral_weight > hori_lateral_max_weighted_distance:
            hori_lateral_max_weighted_distance = lateral_weight
            hori_lateral_closest_top_segment_index = top_roi_coords_x_cycle.current_index
            hori_lateral_thickness_px = lateral_distance
            hori_top_lateral_point = get_closest_point2d(p1 = hori_lateral_bottom_coord, p2 = p2.coordinates, p3 = p2.coordinates)
            print(hori_top_lateral_point)
            continue

        elif medial_weight > hori_medial_max_weighted_distance:
            hori_medial_max_weighted_distance = medial_weight
            hori_medial_closest_top_segment_index = top_roi_coords_x_cycle.current_index
            hori_medial_thickness_px - medial_distance
            hori_top_medial_point = get_closest_point2d(p1 = hori_medial_bottom_coord, p2 = p2.coordinates, p3 = p3.coordinates)
            continue


        
        



    # TODO: for the shortest line this will be the thickness, calculate the intercepting point for visualization
    # TODO: draw line to image

def hori_csa():
    '''
    Calculates hori csa of image roi
    '''

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

        trimmed_roi_lobf_mid_x = roi_midpoint_from_algo(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        )

        _ = get_hori_polygon(
            roi_coords_x=rotated_roi_coords[:,0], # every column of the first row
            roi_coords_y=rotated_roi_coords[:,1],
            roi_mid_x=trimmed_roi_lobf_mid_x
        )

        fig ,ax = plt.subplots()      
        # fig.add_subfigure(results["img"])
        ax.imshow(rotated_image)
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_lobf_mid_x"])

        colors = ["red", "green", "blue" ,"purple"]

        # print(f'Central av thickness: {results["lisee_lateral_average_thickness_mm"]}')


        

        plt.show()
