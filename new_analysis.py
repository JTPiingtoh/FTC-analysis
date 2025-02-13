import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box,intersection
from shapely.plotting import plot_line, plot_polygon
from typing import Literal

# TODO: needs to be able to take user input px/mm ratio
def mm_to_pixels(mm) -> float:   
    return ((mm / 10) * 454)/3.5

# TODO: define lisee zone constants
class LISEE_ZONE():
    """FTC measurement zones"""
    def MEDIAL(image_width: float|int): 
        return - 0.5 * (1 / 3.0) * image_width
    
    def LATERAL(image_width: float|int): 
        return 0.5 * (1 / 3.0) * image_width

# Defines hori zones as per mm to pixel conversion
# TODO: Is mixing a function with a class good design?
class HORI_ZONE():
    
    MEDIAL: float = - mm_to_pixels(10)
    LATERAL: float = mm_to_pixels(10)
    
print(HORI_ZONE.MEDIAL)

# returns polygon representing roi with 12.5% of roi removed from each end
def trim_roi_polygon(
        roi: ImagejRoi,
        image_height: int|float,
) -> Polygon:
    
    left = roi.left
    right = roi.right
    roi_width = left - right
    new_roi_left = left - (roi_width * 0.125) # Left and right are seemingly flipped
    rew_roi_right = right + (roi_width * 0.125)
    trim_box = box(new_roi_left, 0, rew_roi_right, image_height)

    return intersection(trim_box, Polygon(roi.integer_coordinates)) # Intersect betweeb trim box and roi


def calculated_medium_x(
        roi_polygon: Polygon
) -> tuple[tuple[int,int], bool]:
    ''' 
Finds central point by attempting to find closest inflection points. 
If abnormal inflection points are detected, anaylsis still continues, but image is flagged for review.
Returns central coordinate, and flag bool.
'''
    # adapted from https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python
    # TODO: Find list of coords with negative curve
    roi_exterior = roi_polygon.exterior
    coords = np.asarray(roi_exterior.coords.xy)
    
    print(np.diff(coords))

    return "switches"



# returns 3 shapely box polygons for visualizing Lisee zones
def lisee_CSA_polygon_function(
        image_height: float|int,
        image_width: float|int,
        roi: ImagejRoi
        # ROI zones
) -> tuple[Polygon, Polygon, Polygon]:
    
    medial: float = LISEE_ZONE.MEDIAL(image_width)
    lateral: float = LISEE_ZONE.LATERAL(image_width)

# Return lisee csa measures
def lisee_CSA_measures(
        image_array: ArrayLike, 
        roi: ImagejRoi
        # TODO: ROI zones
        ) -> tuple[float, float, float]:
    pass


# main function for FTC analysis
def FTC_analysis(
        image_array: ArrayLike, 
        roi: ImagejRoi
        ) -> dict:


    # Generate polygons to visualize Lisee measurement zones
    

    # medium_x_coord: int = calculate_medium_x()

    lisee_CSA_polygons: tuple[Polygon, Polygon, Polygon] = lisee_CSA_polygon_function(roi=roi)

    # Calculate Lisee CSAs
    lisee_CSAs: tuple[float, float, float] =  lisee_CSA_measures(image_array=image_array, roi=roi) 

    

    # Calcualte Lisee EIs

    # Calcualte Hori thicknesses

    # Calculate Hori CSA

    pass

'''
For testing only
'''
if __name__ == "__main__":
    with TiffFile('502 with roi.tif') as tif:
        
        image = tif.pages[0].asarray()
        assert tif.imagej_metadata is not None
        overlays = tif.imagej_metadata['ROI']
        roi = ImagejRoi.frombytes(overlays)
        coords = roi.integer_coordinates
        left = roi.left
        top = roi.top      
        coords += [left, top]

        fig ,ax = pyplot.subplots()      
        ax.imshow(image)
        ax.plot(coords[:, 0], coords[:, 1])
        trimmed_poly: Polygon = trim_roi_polygon(roi=roi, image_height=image.shape[0])


        # coords = np.asarray(trimmed_poly.exterior.coords.xy)
        # # print(coords)
        # gradient = np.gradient(coords)

        print(f"switches: {calculated_medium_x(trimmed_poly)}")

        plot_polygon(trimmed_poly, ax=ax, color="red")
        pyplot.show()