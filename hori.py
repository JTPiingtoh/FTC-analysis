import numpy as np
from PIL import Image

from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point, contains_xy
from shapely.plotting import plot_line, plot_polygon, plot_points
import matplotlib.pyplot as plt

from midpoint_from_algo import roi_midpoint_from_algo
from roi_rotate import roi_leftmost_rightmost, rotate_image_and_roi
from roi_trim import trim_roi_coords_roi_based
from conversions import mm_to_pixels, pixels_to_mm


# Defines hori zones as per mm to pixel conversion
# TODO: Is mixing a function with a class good design?
class HORI_ZONE():
    
    MEDIAL: float = - mm_to_pixels(10)
    LATERAL: float = mm_to_pixels(10)

def get_hori_polygon(
    image_height: float|int,
    image_width: float|int,
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    roi_mid_x: float|int       
):
    hori_lateral_x = roi_mid_x - HORI_ZONE.LATERAL
    hori_medial_x = roi_mid_x - HORI_ZONE.MEDIAL
    # TODO: iteratre through hori xs
    # TODO: iteratre through top line
    # TODO: add the indexes of each pair of top line points and the shortest distance bwetween the point
    # TODO: for the shortest line this will be the thickness, calculate the intercepting point for visualization
    # TODO: draw line to image

def hori_csa():
    '''
    Calculates hori csa of image roi
    '''

if __name__ == "__main__":
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

        fig ,ax = plt.subplots()      
        # fig.add_subfigure(results["img"])
        ax.imshow(rotated_image)
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_lobf_mid_x"])

        colors = ["red", "green", "blue" ,"purple"]

        # print(f'Central av thickness: {results["lisee_lateral_average_thickness_mm"]}')


        

        plt.show()
