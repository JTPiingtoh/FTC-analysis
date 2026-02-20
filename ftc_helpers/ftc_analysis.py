import logging

import numpy as np
from PIL import Image

from numpy.typing import ArrayLike
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely.plotting import plot_polygon

import matplotlib.pyplot as plt

from ftc_helpers.midpoint_lobf import roi_midpoint_lobf
from ftc_helpers.midpoint_from_algo import roi_midpoint_from_algo
from ftc_helpers.roi_rotate import rotate_image_and_roi
from ftc_helpers.roi_trim import trim_roi_coords_roi_based
from ftc_helpers.conversions import pixels_to_mm

from ftc_helpers.hori import get_hori_csa_px, get_hori_central_thickness_px, get_hori_coords_thickness
from ftc_helpers.lisee import lisee_CSA_polygon_function, lisee_zone_average_thickness
from ftc_helpers.lisee_EI import get_average_ei

logger = logging.getLogger("__main__")

# BUG: Trim factors above 0.25 appears to break midpoint_lobf, which in turn
# breaks lisee_roi_polygon, likely due to too much of the of the roi being trimmed.
# For trim factors greater than 0.25, algorithic detection of midpoint will be needed instead.

# TODO: needs to be able to take user input px/mm ratio

# main function for FTC analysis
def FTC_analysis(
        image_array: ArrayLike, 
        roi: ImagejRoi,
        filename: str,
        **kwargs
        ) -> dict:

    # convert to pil image for rotation late
    image = Image.fromarray(image_array)
    image_width = image.width
    image_height = image.height
    
    roi_coords = roi.integer_coordinates
    
    left = roi.left
    top = roi.top      
    right = roi.right
    bottom = roi.bottom

    roi_coords += [left, top]

    rotated_image, rotated_roi_coords = rotate_image_and_roi(image=Image.fromarray(image_array), roi=roi)

    trimmed_roi_coords = trim_roi_coords_roi_based(
    roi_coords_x=rotated_roi_coords[:,0], # every column of the first row
    roi_coords_y=rotated_roi_coords[:,1],
    roi_left=left,
    roi_right=right,
    image_height=image_height,
    trim_factor=0.25
    )

    # TODO: add check for midpoint#
    # FOR LOGGING ONLY
    trimmed_roi_lobf_mid_x = roi_midpoint_lobf(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        polynomial_order=4
    )

    trimmed_roi_algo_mid_x = roi_midpoint_from_algo(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1]
    )

    # if abs(trimmed_roi_lobf_mid_x - trimmed_roi_algo_mid_x) / (abs(left - right)) > 0.1:
    #     logger.info(f"Review {filename}. Program found potential discrepency with ROI midpoint.")

    logger.info(f"lobf / algo {abs(trimmed_roi_lobf_mid_x - trimmed_roi_algo_mid_x) / (abs(left - right))}")

    hori_coords, hori_ML_thicknesses = get_hori_coords_thickness(
        roi_coords_x=trimmed_roi_coords[0], # every column of the first row
        roi_coords_y=trimmed_roi_coords[1],
        roi_mid_x=trimmed_roi_algo_mid_x,
        return_thickness_px=True
    )

    hori_medial_thickness_px, hori_lateral_thickness_px = hori_ML_thicknesses

    hori_csa_px = get_hori_csa_px(
        image_height=image_height,
        image_width=image_width,
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        hori_coords=hori_coords,
        roi_mid_x=trimmed_roi_algo_mid_x,
    )

    hori_central_thickness_px = get_hori_central_thickness_px(
        image_height=image_height,
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        roi_mid_x=trimmed_roi_algo_mid_x,   
        )

    results_dict = {}

    results_dict["Image name"] = filename

    results_dict["Hori_medial_thickness_mm"] = pixels_to_mm(hori_medial_thickness_px)
    results_dict["Hori_lateral_thickness_mm"] = pixels_to_mm(hori_lateral_thickness_px)
    results_dict["Hori_central_thickness_mm"] = pixels_to_mm(hori_central_thickness_px)
    results_dict["Hori_csa_mm"] = hori_csa_px * (pixels_to_mm(1) ** 2)

    lisee_polygons = lisee_CSA_polygon_function(
        image_height=image_height,
        image_width=image_width,
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        roi_mid_x=trimmed_roi_algo_mid_x   
    )

    lisee_lateral_roi_polygon, lisee_central_roi_polygon, lisee_medial_roi_polygon = lisee_polygons
    lisee_lateral_average_thickness_pixles = lisee_zone_average_thickness(lisee_zone=lisee_lateral_roi_polygon)
    lisee_central_average_thickness_pixles = lisee_zone_average_thickness(lisee_zone=lisee_central_roi_polygon)
    lisee_medial_average_thickness_pixles = lisee_zone_average_thickness(lisee_zone=lisee_medial_roi_polygon)

    results_dict["Lisee_Medial_average_mm_thickness"]  =     pixels_to_mm(lisee_medial_average_thickness_pixles)
    results_dict["Lisee_Lateral_average_mm_thickness"] =     pixels_to_mm(lisee_lateral_average_thickness_pixles)
    results_dict["Lisee_Intercondyl_average_mm_thickness"] = pixels_to_mm(lisee_central_average_thickness_pixles)

    results_dict["lisee_medial_pixels"] = lisee_medial_roi_polygon.area
    results_dict["lisee_lateral_pixels"] = lisee_lateral_roi_polygon.area
    results_dict["lisee_central_pixels"] = lisee_central_roi_polygon.area

    results_dict["lisee_medial_area_mm"] = lisee_medial_roi_polygon.area * (pixels_to_mm(1) ** 2)
    results_dict["lisee_lateral_area_mm"] = lisee_lateral_roi_polygon.area * (pixels_to_mm(1) ** 2)
    results_dict["lisee_central_area_mm"] = lisee_central_roi_polygon.area * (pixels_to_mm(1) ** 2)

    results_dict["lisee_medial_average_thickness_pixles"]  = lisee_medial_average_thickness_pixles 
    results_dict["lisee_lateral_average_thickness_pixles"] = lisee_lateral_average_thickness_pixles
    results_dict["lisee_central_average_thickness_pixles"] = lisee_central_average_thickness_pixles

    # TODO: add echo intensity
    rotated_image_array = np.array(rotated_image)
    third: float = 1.0 / 3.0

    rotated_image_array_gryscl = np.dot(rotated_image_array[..., :3], [third,third,third]) # remove alpha, dot product rgb channels
    rotated_image_array_gryscl_width: int = rotated_image_array_gryscl.shape[0]
    rotated_image_array_gryscl_height: int = rotated_image_array_gryscl.shape[1]

    results_dict["lisee_medial_ei"] = get_average_ei(
        image_width=rotated_image_array_gryscl_width,
        image_height=rotated_image_array_gryscl_height,
        polygon=lisee_medial_roi_polygon,
        image_array=rotated_image_array_gryscl)

    results_dict["lisee_lateral_ei"] = get_average_ei(
        image_width=rotated_image_array_gryscl_width,
        image_height=rotated_image_array_gryscl_height,
        polygon=lisee_lateral_roi_polygon,
        image_array=rotated_image_array_gryscl)

    results_dict["lisee_central_ei"] = get_average_ei(
        image_width=rotated_image_array_gryscl_width,
        image_height=rotated_image_array_gryscl_height,
        polygon=lisee_central_roi_polygon,
        image_array=rotated_image_array_gryscl)

    # visualize image
    plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(rotated_image)
    colors = ["red", "green", "blue"]
    for i, polygon in enumerate(lisee_polygons):
            plot_polygon(polygon=polygon, ax=ax, color=colors[i])

    return results_dict, fig
    

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


        trimmed_roi_coords = trim_roi_coords_roi_based(
        roi_coords_x=rotated_roi_coords[:,0], # every column of the first row
        roi_coords_y=rotated_roi_coords[:,1],
        roi_left=left,
        roi_right=right,
        image_height=image_height,
        trim_factor=0.25
        )

        trimmed_roi_lobf_mid_x = roi_midpoint_lobf(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        polynomial_order=4
        )

        lisee_polygons = lisee_CSA_polygon_function(
         image_height=image_height,
         image_width=image_width,
         roi_coords_x=trimmed_roi_coords[0],
         roi_coords_y=trimmed_roi_coords[1],
         roi_mid_x=trimmed_roi_lobf_mid_x   
        )

        
        fig ,ax = plt.subplots()      
        # fig.add_subfigure(results["img"])
        ax.imshow(rotated_image)
        ax.plot(left, top, 'go')
        ax.plot(right, bottom, 'go')
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_lobf_mid_x"])

        colors = ["red", "green", "blue" ,"purple"]

        # print(f'Central av thickness: {results["lisee_lateral_average_thickness_mm"]}')

        for i, polygon in enumerate(lisee_polygons):
            plot_polygon(polygon=polygon, ax=ax, color=colors[i])

        

        plt.show()
