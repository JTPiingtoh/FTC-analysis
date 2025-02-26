import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point
from shapely.plotting import plot_line, plot_polygon
from typing import Literal
from midpoint_lobf import roi_midpoint_lobf

# TODO: needs to be able to take user input px/mm ratio
def mm_to_pixels(
    mm: float|int,
    pixles_per_mm: float|int = 454
    ) -> float:   
    return ((mm / 10) * pixles_per_mm)/3.5

# TODO: define lisee zone constants
class LISEE_ZONE():
    """FTC measurement zones"""
    def MEDIAL(roi_width: float|int): 
        return - 0.5 * (1 / 3.0) * roi_width
    
    def LATERAL(roi_width: float|int): 
        return 0.5 * (1 / 3.0) * roi_width

# Defines hori zones as per mm to pixel conversion
# TODO: Is mixing a function with a class good design?
class HORI_ZONE():
    
    MEDIAL: float = - mm_to_pixels(10)
    LATERAL: float = mm_to_pixels(10)
    
# print(HORI_ZONE.MEDIAL)

# returns polygon representing roi with 12.5% of roi removed from each end
def trim_roi_polygon(
        roi: ImagejRoi,
        image_height: int|float,
        trim_factor: float
) -> Polygon:
    '''
    Returns coordinates of a trimmed roi
    '''
    roi_width = roi.left - roi.right
    new_roi_left = roi.right - (roi_width * trim_factor) # Left and right are seemingly flipped
    new_roi_right = roi.right + (roi_width * trim_factor)
    trim_box = box(new_roi_left, 0, new_roi_right, image_height)

    return intersection(trim_box, Polygon(roi.integer_coordinates)) # Intersect betweeb trim box and roi


def trim_roi_coords(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    image_width: int|float,
    image_height: int|float,
    trim_factor: float
) -> np.ndarray[np.ndarray, np.ndarray]:
    '''
    Returns ndarray representing coords of trimmed roi based on image width. Ensure roi supplied to function has coords aligned with background image. 
    '''

    new_roi_left: int = image_width * (trim_factor/ 2) # Left and right are seemingly flipped
    new_roi_right: int = image_width - image_width * (trim_factor/ 2)

    trim_box = box(new_roi_left, 0, new_roi_right, image_height)

    intersection_polgyon= intersection(trim_box, Polygon(np.column_stack((roi_coords_x, roi_coords_y))), grid_size=None)

    return np.array(intersection_polgyon.exterior.coords.xy)
    

def box_polygon_intersect(
        box_coords: tuple[float, float, float, float],
        polygon: Polygon
) -> Polygon:
    '''
    Convenience helper function for lisee_CSA_polygon_function. Takes a Polygon and box coordinates,
    producing a polygon representing their overlap
    '''
    xmin, ymin, xmax, ymax = box_coords

    return intersection(box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), polygon)[0]


# returns 3 shapely box polygons for visualizing Lisee zones
def lisee_CSA_polygon_function(
        image_height: float|int,
        image_width: float|int,
        roi_coords_x: np.ndarray,
        roi_coords_y: np.ndarray,
        roi_mid_x: float|int
        
) -> tuple[Polygon, Polygon, Polygon]:
    
    roi_left_x = min(roi_coords_x)
    roi_right_x = max(roi_coords_x)
    roi_width = roi_left_x - roi_right_x

    medial_adjust: float = LISEE_ZONE.MEDIAL(roi_width)
    lateral_adjust: float = LISEE_ZONE.LATERAL(roi_width)

    medial_x = roi_mid_x + medial_adjust
    lateral_x = roi_mid_x + lateral_adjust

    trimmed_roi_polygon = Polygon(np.column_stack((roi_coords_x, roi_coords_y)))

    medial_roi_polygon = box_polygon_intersect(
        box_coords=(roi_left_x, 0, lateral_x, image_height),
        polygon=trimmed_roi_polygon)

    central_roi_polygon = box_polygon_intersect(
        box_coords=(lateral_x, 0, medial_x, image_height),
        polygon=trimmed_roi_polygon
    )

    lateral_roi_polygon = box_polygon_intersect(
        box_coords=(medial_x, 0, roi_right_x, image_height),
        polygon=trimmed_roi_polygon
    )
    
    assert isinstance(medial_roi_polygon, Polygon)

    print(medial_x)

    return lateral_roi_polygon, central_roi_polygon, medial_roi_polygon

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
        roi: ImagejRoi,
        **kwargs
        ) -> dict:

    results_dict = {}

    image_width = image_array.shape[1]
    image_height = image_array.shape[0]
    
    roi_coords = roi.integer_coordinates

    trimmed_roi_coords = trim_roi_coords(
        roi_coords_x=roi_coords[:,0], # every column of the first row
        roi_coords_y=roi_coords[:,1],
        image_width=image_width,
        image_height=image_height,
        trim_factor=0.25)

    results_dict["trimmed_roi_coords"] = trimmed_roi_coords

    trimmed_roi_mid_x = roi_midpoint_lobf(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        polynomial_order=7
    )

    results_dict["trimmed_roi_mid_x"] = trimmed_roi_mid_x

    lisee_polygons = lisee_CSA_polygon_function(
         image_height=image_height,
         image_width=image_width,
         roi_coords_x=trimmed_roi_coords[0],
         roi_coords_y=trimmed_roi_coords[1],
         roi_mid_x=trimmed_roi_mid_x   
        )

    results_dict["lisee_polygons"] = lisee_polygons

    lisee_lateral_roi_polygon, lisee_central_roi_polygon, lisee_medial_roi_polygon = lisee_polygons

    results_dict["lisee_lateral_pixels"] = lisee_lateral_roi_polygon.area
    results_dict["lisee_central_pixels"] = lisee_central_roi_polygon.area
    results_dict["lisee_medial_pixels"] = lisee_medial_roi_polygon.area

    results_dict["lisee_lateral_area_mm"] = mm_to_pixels(lisee_lateral_roi_polygon.area)
    results_dict["lisee_central_area_mm"] = mm_to_pixels(lisee_central_roi_polygon.area)
    results_dict["lisee_medial_area_mm"] = mm_to_pixels(lisee_medial_roi_polygon.area)

    grayscale_img_arr = np.dot(image_array[..., :3], [1, 1, 1]) # remove alpha

    for x in range(grayscale_img_arr.shape[0]):
        for y in range(grayscale_img_arr.shape[1]):

            if Point(x, y).within(lisee_lateral_roi_polygon):
                pass

    return results_dict
    

    # Calcualte Lisee EIs

    # Calcualte Hori thicknesses

    # Calculate Hori CSA



'''
For testing only
'''
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


        
        results = FTC_analysis(image,roi)
        trimmed_roi_polygon = Polygon(np.column_stack(results["trimmed_roi_coords"]))
        


        fig ,ax = pyplot.subplots()      
        ax.imshow(image)
        ax.plot(coords[:, 0], coords[:, 1])
        ax.plot(left, top, 'go')
        ax.plot(right, bottom, 'go')
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_mid_x"])

        colors = ["red", "green", "blue"]

        for i, polygon in enumerate(results["lisee_polygons"]):
            plot_polygon(polygon=polygon, ax=ax, color=colors[i])

        pyplot.show()