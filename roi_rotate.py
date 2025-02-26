import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point
from shapely.plotting import plot_line, plot_polygon
from typing import Literal


def roi_leftmost_rightmost(
    left: int,
    right:int,
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray
): #-> tuple[tuple[float, float]]:
    '''
    Returns the leftmost and rightmost roi coords, with a preference for coords of higher lower y values
    (higher up on the image)
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
        leftmost_cord = leftmost_coords[0]

    else:
        leftmost_cord = leftmost_coords[leftmost_coords[1,:] == min(leftmost_coords[1,:])]
        print(leftmost_cord)


    return 0

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

        print(roi_leftmost_rightmost(
            left=left, 
            right=right,
            roi_coords_x=coords[:,0], # every column of the first row
            roi_coords_y=coords[:,1])
        )



        fig ,ax = pyplot.subplots()      
        ax.imshow(image)
        ax.plot(coords[:, 0], coords[:, 1])
        ax.plot(left, top, 'go')
        ax.plot(right, bottom, 'go')
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_mid_x"])

        colors = ["red", "green", "blue"]

    

        pyplot.show()