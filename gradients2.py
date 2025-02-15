import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box,intersection, LineString
from shapely.plotting import plot_line, plot_polygon
from typing import Literal

#TODO: return 2 from roi top line that intersect a ertical line

def intersecting_segment_coords(      
    poly_coords: np.ndarray,
    line: LineString,
    image_height: float|int
):
    '''
    Returns 2 points of a polygon that, when a line is drawn between 
    them, interects another straight line,
    '''
    
    min_y = image_height
    point_index = 0
    
    for i in range(poly_coords.shape[1] - 1):
        point1 = poly_coords[0][i], poly_coords[1][i]
        point2 = poly_coords[0][i+1], poly_coords[1][i+1]

        poly_segment = LineString([point1, point2])

        intersect = intersection(line, poly_segment)

        if intersect:
       
            intersect_y = intersect.coords.xy[1][0]
            
            if intersect_y < min_y:
                point_index = i
                min_y = intersect_y

    # now that a top line coord is found, we must progress towards the center of the image. if we are starting
    # from the image LHS, we need ensure we go right (increase x) AND ensure that y increases. 
    # First, we probe the next point in the list. If x decreases, we need to go change the index in the other
    # dircetion. Now we have our next point, we then need to see if y increases. If it has, we carry on. If it 
    # hasnt, we have reached our destination.

    i = 1
    current_x = poly_coords[0][point_index]
    next_x = poly_coords[0][point_index + i]
    current_y = poly_coords[1][point_index]
    next_y = poly_coords[1][point_index + i]
    # if LHS, ensure m < 0
    dx = next_x - current_x
    dy = next_y - current_y
    m = dy / dx

    if m > 0: # if m is positive
        # TODO: needs to contract line 
        pass

    if next_x < current_x: # TODO: this needs to be inverted when coming from RHS
        i = -1

    Y_INCREASING = True

    while Y_INCREASING:

        current_y = poly_coords[1][point_index]
        next_y = next_x = poly_coords[1][point_index + i]
        
        if next_y > current_y:
            point_index += i
            continue

        Y_INCREASING = False
        break


    print(poly_coords[0][point_index], poly_coords[1][point_index])
    
    
   

    return 0
    
def find_mid_x(
    polygon: Polygon,
    image_height: float|int,
    image_width: float|int
    ):

    coords = np.asarray(polygon.coords.xy)
    roi_center = image_width / 2
    lhs = roi_center + image_width

    while(True):

        lhs_line = LineString([[image_width / 7 , 0], [image_width / 7, image_height]])

    intersecting_segment_coords(coords=coords, line=line, image_height=image_height)
    # LHS
    # Set line width
    # Return 2 points that intersect line, and their index within the coords
    # compute m
    # if m is positive, contract line width
    # else iterate through points until a low point is found

    # repeat on RHS
    
    
    pass


if __name__ == "__main__":
    with TiffFile('504 with roi.tif') as tif:
        
        image = tif.pages[0].asarray()
        assert tif.imagej_metadata is not None
        overlays = tif.imagej_metadata['ROI']
        roi = ImagejRoi.frombytes(overlays)
        coords = roi.integer_coordinates
        left = roi.left
        top = roi.top      
        coords += [left, top]
        polygon = Polygon(coords)

        height = image.shape[0]
        width = image.shape[1]
        line = LineString([[width / 7 , 0], [width / 7, height]])
        
        intersecting_segment_coords(polygon, line, height)


        fig ,ax = plt.subplots()      
        ax.imshow(image)
        plot_polygon(polygon=polygon, ax=ax)
        plot_line(line=line, ax=ax)
        intersect = intersection(line, polygon)
        plot_line(intersect, ax=ax)
        plt.show()
        