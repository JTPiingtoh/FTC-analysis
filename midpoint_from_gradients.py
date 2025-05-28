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


'''
Finds the roi midpoint using an algorithm
'''

def intersecting_segment_coords(      
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    line: LineString,
    # image_height: float|int
) -> int:
    '''From a list of coords, returns index of the coordinate with the lowest y value of ROI points that intersect 
    a (vertical) line'''

    assert roi_coords_x.shape == roi_coords_y.shape

    max_y = roi_coords_y.max()
    print(f"Initial max_y: {max_y}")
    point_index = 0
    

    for i in range(roi_coords_x.shape[0] - 1):
        point1 = roi_coords_x[i], roi_coords_y[i]
        point2 = roi_coords_x[i+1], roi_coords_y[i+1]

        poly_segment = LineString([point1, point2])

        intersect = intersection(line, poly_segment)

        if intersect:
       
            intersect_y = intersect.coords.xy[1][0]

            print(f"intersect_y: {intersect_y}")
            
            if intersect_y < max_y:
                point_index = i
                max_y = intersect_y


    print(f"Final max_y: {max_y}")
    


    return point_index


def roi_midpoint_from_gradients(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray
):
    # Ensure trim #
        
    assert roi_coords_x.shape == roi_coords_y.shape

    roi_left_x = float(min(roi_coords_x))
    roi_right_x = float(max(roi_coords_x))

    roi_top_x = float(min(roi_coords_y))
    roi_bottom_x = float(max(roi_coords_y))
    roi_width = float(roi_right_x - roi_left_x)


    # Find topline using vertical ray from roi midpoint #         
    roi_halfway_vertical_line = LineString(
        [[roi_left_x + (roi_width/2.0), roi_bottom_x ], 
        [roi_left_x + (roi_width/2.0), roi_top_x]]
    )

    top_line_intersect_index = intersecting_segment_coords(
        roi_coords_x=roi_coords_x,
        roi_coords_y=roi_coords_y,
        line=roi_halfway_vertical_line
    )

    print(roi_coords_x[top_line_intersect_index], roi_coords_y[top_line_intersect_index])

    iterator = 1
    edges = 0

    max_y = 0
    med_x = []
    
    roi_coords_y_cycle = CycleLoop(iterable=roi_coords_y, start_index=top_line_intersect_index)
    roi_coords_x_cycle = CycleLoop(iterable=roi_coords_x, start_index=top_line_intersect_index)

    # Iterate through top line to find y coords
    while edges < 2:


    iterator = 1
    edges = 0

    max_y = 0
    med_x = []

    # Iterate through top line, finding the lowest
    
    # TODO: handle straight lines
    while edges < 2:

        current_x = roi_coords_x_cycle.current
        next_x = roi_coords_x_cycle.peek1(iterator)

        current_y = roi_coords_y[roi_coords_x_cycle.current_index]
        if current_y > max_y:
            max_y = current_y
            med_x = roi_coords_x_cycle.current 

        if next_x == current_x:
            iterator *=- 1
            edges += 1

        roi_coords_x_cycle.step(iterator)

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

        trimmed_roi_mid_x = roi_midpoint_lobf(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        polynomial_order=4
        )

        midpoint = roi_midpoint_from_gradients(
            roi_coords_x=trimmed_roi_coords[0],
            roi_coords_y=trimmed_roi_coords[1],            
        )

        poly_midpoint = roi_midpoint_lobf(
            roi_coords_x=trimmed_roi_coords[0],
            roi_coords_y=trimmed_roi_coords[1]
        )


        print(f"midpoint: {midpoint}")
        print(f"poly_midpoint: {poly_midpoint}")


        fig, ax = plt.subplots()
        # ax.imshow(rotated_image)
        ax.plot(trimmed_roi_coords[0], trimmed_roi_coords[1])
        plt.show()

    