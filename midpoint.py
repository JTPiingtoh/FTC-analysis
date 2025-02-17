import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box,intersection, LineString
from shapely.plotting import plot_line, plot_polygon
from typing import Literal
import multipolyfit as mpf


def intersecting_segment_coords(      
    polygon: Polygon,
    line: LineString,
    image_height: float|int
) -> int:
    '''From a list of coords, returns index of the coordinate with the lowest y value of ROI points that intersect 
    a (vertical) line'''

    poly_coords = np.asarray(polygon.exterior.xy)
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


    return point_index


def gradient(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    '''Gradient between 2 points'''
    assert point1.shape == (2,)
    assert point2.shape == (2,)
    
    x1, y1 = point1
    x2, y2 = point2

    if x2 - x1 == 0: # 0/0

        if y2 > y1:
            return np.inf
        elif y2 < y1:
            return -np.inf
        else: #dy and dx are both zero, therefore the 2 points are the same.
            return 0 # Treat gradient as 0, as points are at the same heigh

    else:
        return (y2-y1)/(x2-x1) # dy/dx


def roi_width(
    coords: np.ndarray,
)-> float:
    '''Returns the halfway point of the roi width'''
    
    x_coords = coords[:,0]

    return ((max(x_coords) - min(x_coords))  ) 


def roi_vert_xs(
    roi_width: float|int   
):
    '''
    Returns a list of factors by which to multiply the roi width 
    '''
    arr = np.linspace(0.1, 0.9, 10)
    arr *= (roi_width / 2)

    return arr
    


def correct_direction(
    point1: np.ndarray,
    point2: np.ndarray,
    direction: Literal["l, r"] = "l"
) -> bool:
    ''' Takes 2 points and a direction, and checks if point 2 is [direction]
    of point1. Returns false if x is the same, however, as gradient check comes first
    this should be fine'''  

    assert point1.shape == (2,)
    assert point2.shape == (2,)
    
    if direction == "l":
        return point2[0] > point1[0] 

    else:
        return point2[0] < point1[0] 


def generate_vertical_intercept_line_xs():
    pass


def find_midpoint(
    roi_polygon: Polygon
):
    LEFT = "l"
    RIGHT = "r"

    


    # define lines
    directions = [LEFT, RIGHT]
    
    x_low_points = []
    for direction in directions:
        
        vertical_intercept_line_xs = generate_vertical_intercept_line_xs(
            roi_width=roi_width,

        )


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
        line = LineString([[width / 1.2 , 0], [width / 1.2, height]])
        
        index = intersecting_segment_coords(polygon, line, height)

        line_of_best_fit = np.interp(range(0, width) ,coords[0], coords[1])

        print(line_of_best_fit)

        print(gradient(coords[index], coords[index+1]))
        print(coords[index], coords[index+1])

        print(correct_direction(coords[index], coords[index+1], "r"))
        roi_width = roi_width(coords=coords)

        xs = roi_vert_xs(roi_width=roi_width)
        xs += left # adjust for roi offset
        fig ,ax = plt.subplots()      
        ax.imshow(image)
        plot_polygon(polygon=polygon, ax=ax)
        plot_line(line=line, ax=ax)


        ax.plot(line_of_best_fit)
        ax.axvline((roi_width/2) + left)
        ax.vlines(xs, 0, width)
        intersect = intersection(line, polygon)
        plot_line(intersect, ax=ax)
        plt.show()
        