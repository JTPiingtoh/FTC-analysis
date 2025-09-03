import numpy as np
from PIL import Image
from itertools import cycle
from numpy.typing import ArrayLike
from matplotlib import pyplot
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point, contains_xy, LineString
from shapely.plotting import plot_line, plot_polygon, plot_points
from typing import Literal
import matplotlib.pyplot as plt
from swingloop import CycleLoop

from midpoint_lobf import roi_midpoint_lobf
from helpers import intersecting_segment_coords, line_start_index


def roi_midpoint_from_algo(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray
):
    '''
    Figures out roi by finding lowest y point on the top line of the roi. This function is called assuming that
    the roi has been trimmed, and therefore has 2 vertical lines on both sides of the trimmed roi.
    '''   

    top_line_intersect_index = line_start_index(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        intersect_loc="top"
    )

    roi_coords_x_cycle = CycleLoop(iterable=roi_coords_x, start_index=top_line_intersect_index)
    roi_coords_y_cycle = CycleLoop(iterable=roi_coords_y, start_index=top_line_intersect_index)

    iterator = 1
    edges = 0
    max_y = 0
    med_x = 0
    iterations = 0
    MAX_ITERATIONS = 200

    # for if 2 points have the same 
    flat_pairs = []

    # TODO: handle straight lines
    while edges < 2:

        roi_coords_x_cycle.step(iterator)
        roi_coords_y_cycle.step(iterator)
        iterations += 1

        if iterations > MAX_ITERATIONS:
            raise RuntimeError(f"roi_mid_point_from_gradients() could not find midpoint after {MAX_ITERATIONS} iterations." \
                               "Roi may not have been trimmed.")
        
        current_x = roi_coords_x_cycle.current
        next_x = roi_coords_x_cycle.peek1(iterator)

        if next_x == current_x:
            iterator *=- 1
            edges += 1
            continue
            
        current_y = roi_coords_y_cycle.current
        if current_y > max_y:
            max_y = current_y
            med_x = roi_coords_x_cycle.current 

        next_y = roi_coords_y_cycle.peek1(iterator)
        if current_y == next_y:            
            flat_pairs.append(((current_x, current_y), (next_x, next_y)))

        

    if len(flat_pairs) > 0:

        flat_pairs = np.array(flat_pairs)

        # finds the flat pair of points with the highest y values, and averages
        # their x coords if their y value is greater or eq to max_y
        max_flat_y = np.max(flat_pairs[:,0,1])
        if max_flat_y >= max_y:
            med_x = np.mean(flat_pairs[flat_pairs[:,0,1] == max_flat_y][0].T[0])
        
    return med_x



if __name__ == "__main__":

    from roi_rotate import rotate_image_and_roi
    from roi_trim import trim_roi_coords_roi_based

    with TiffFile('504 with roi.tif') as tif:
    
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


        algo_mid_x = roi_midpoint_from_algo(
            roi_coords_x=trimmed_roi_coords[0],
            roi_coords_y=trimmed_roi_coords[1],            
        )

        poly_midpoint = roi_midpoint_lobf(
            roi_coords_x=trimmed_roi_coords[0],
            roi_coords_y=trimmed_roi_coords[1]
        )


        print(f"midpoint: {algo_mid_x}")
        print(f"poly_midpoint: {poly_midpoint}")


        fig, ax = plt.subplots()
        # ax.imshow(rotated_image)
        ax.plot(trimmed_roi_coords[0], trimmed_roi_coords[1])
        plt.show()

    