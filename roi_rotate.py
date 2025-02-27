import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point
from shapely.plotting import plot_line, plot_polygon
from typing import Literal
import math

def roi_leftmost_rightmost(
    left: int,
    right:int,
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray
)-> tuple[np.ndarray, np.ndarray]:
    '''
    Takes the left and right coordinates of a roi bounding box, roi x coordinates and roi y coordinates.
    Returns the leftmost and rightmost roi coords, with a preference for coords of higher lower y values
    (higher up on the image), as a tuple of ndarrays of shape (2,) representing 2 coordinates.
    '''

    if len(roi_coords_x) != len(roi_coords_y):
        raise IndexError

    leftmost_coords = []
    rightmost_coords = []

    for i in range(len(roi_coords_x)):
        if roi_coords_x[i] == left:
            leftmost_coords.append((roi_coords_x[i], roi_coords_y[i]))
        elif roi_coords_x[i] == right:
            rightmost_coords.append((roi_coords_x[i], roi_coords_y[i]))


    if not leftmost_coords or not rightmost_coords:
        raise RuntimeError("Failed to match roi coordinates with ImageJRoi.integercoordinates.left")
    else:
        leftmost_coords = np.array(leftmost_coords)
        rightmost_coords = np.array(rightmost_coords)
    
    
    if len(leftmost_coords) == 1:
        leftmost_coord = leftmost_coords[0]
    else:
        leftmost_coord = leftmost_coords[leftmost_coords[:,1] == min(leftmost_coords[:,1])][0]
        
    if len(rightmost_coords) == 1:
        rightmost_coord = rightmost_coords[0]
    else:
        rightmost_coord = rightmost_coords[rightmost_coords[:,1] == min(rightmost_coords[:,1])][0]
    
    assert leftmost_coord.shape == rightmost_coord.shape == (2,)

    return (leftmost_coord, rightmost_coord)


if __name__ == "__main__":
    with TiffFile('504 with roi.tif') as tif:
        
        image = tif.pages[0].asarray()

        image_width = image.shape[1]
        image_height = image.shape[0]
        assert tif.imagej_metadata is not None
        overlays = tif.imagej_metadata['ROI']
        roi = ImagejRoi.frombytes(overlays)
        coords = roi.integer_coordinates
        left = roi.left
        top = roi.top      
        right = roi.right
        bottom = roi.bottom

        coords += [left, top]

        leftmost_coord, rightmost_coord = roi_leftmost_rightmost(
            left=left, 
            right=right,
            roi_coords_x=coords[:,0], # every column of the first row
            roi_coords_y=coords[:,1])
        

        dy = rightmost_coord[1] - leftmost_coord[1]
        dx = rightmost_coord[0] - leftmost_coord[0]

        angle_rad = np.arctan2(dy, dx) # dy, dx
        angle_deg = np.rad2deg(angle_rad)
        
        rot_matrix = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)],
                              [np.sin(-angle_rad), np.cos(-angle_rad)]], dtype='float64')

        # coords = np.array([coords[:,0], coords[:,1]], dtype='float64')
        
        print(coords)
        print(rot_matrix)
        coords = np.dot(coords, rot_matrix.T)


        fig ,ax = pyplot.subplots()      
        ax.imshow(image)
        ax.plot(coords[:, 0], coords[:, 1])
        ax.plot(left, top, 'go')
        ax.plot(right, bottom, 'go')
        ax.plot([leftmost_coord[0], rightmost_coord[0]], [leftmost_coord[1], rightmost_coord[1]])
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_mid_x"])

        colors = ["red", "green", "blue"]

    

        pyplot.show()