import numpy as np
from PIL import Image

from numpy.typing import ArrayLike
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box, intersection, Point, contains_xy
from shapely.plotting import plot_line, plot_polygon, plot_points
from typing import Literal
import matplotlib.pyplot as plt
import math

from midpoint_lobf import roi_midpoint_lobf
from midpoint_from_algo import roi_midpoint_from_algo
from roi_rotate import roi_leftmost_rightmost, rotate_image_and_roi
from roi_trim import trim_roi_coords_roi_based
from conversions import mm_to_pixels, pixels_to_mm

# BUG: Trim factors above 0.25 appears to break midpoint_lobf, which in turn
# breaks lisee_roi_polygon, likely due to too much of the of the roi being trimmed.
# For trim factors greater than 0.25, algorithic detection of midpoint will be needed instead.

# TODO: needs to be able to take user input px/mm ratio


    
# TODO move lisee implementations to new file
class LISEE_ZONE():
    """FTC measurement zones"""
    def MEDIAL(roi_width: float|int): 
        return - 0.5 * (1 / 3.0) * roi_width
    
    def LATERAL(roi_width: float|int): 
        return 0.5 * (1 / 3.0) * roi_width


    
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

    # print(new_roi_left)

    return intersection(trim_box, Polygon(roi.integer_coordinates)) # Intersect betweeb trim box and roi


def box_polygon_intersect(
        box_coords: tuple[float, float, float, float],
        polygon: Polygon
) -> Polygon:
    '''
    Convenience helper function for lisee_CSA_polygon_function. Takes a Polygon and box coordinates,
    producing a polygon representing their overlap
    '''
    xmin, ymin, xmax, ymax = box_coords

    return intersection(box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), polygon)


# returns 3 shapely box polygons for visualizing Lisee zones
def lisee_CSA_polygon_function(
        image_height: float|int,
        image_width: float|int,
        roi_coords_x: np.ndarray,
        roi_coords_y: np.ndarray,
        roi_mid_x: float|int         
) -> tuple[Polygon, Polygon, Polygon]:
    
    roi_left_x = float(min(roi_coords_x))
    roi_right_x = float(max(roi_coords_x))
    roi_width = float(roi_left_x - roi_right_x)

    medial_adjust: float = LISEE_ZONE.MEDIAL(roi_width)
    lateral_adjust: float = LISEE_ZONE.LATERAL(roi_width)

    medial_x = float(roi_mid_x[0] + medial_adjust)
    lateral_x = float(roi_mid_x[0] + lateral_adjust)

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
    
    assert isinstance(central_roi_polygon, Polygon)
    assert isinstance(medial_roi_polygon, Polygon)
    assert isinstance(lateral_roi_polygon, Polygon)

    return lateral_roi_polygon, central_roi_polygon, medial_roi_polygon## TODO: remove trimmed


def line_length(
    coords_x: np.ndarray,
    coords_y: np.ndarray
)-> float:
    '''
    Takes an ordered series of coordinates, and returns the total length of lines
    drawn between them.
    '''
    if coords_x.size != coords_y.size:
        raise RuntimeError("line_length recieved coordinate arrays if different sizes.")

    if coords_x.ndim != 1 and coords_y.ndim != 1:
        raise RuntimeError(f"line_length takes coord arrays with 1 dimension. coords_x.ndim: {coords_x.ndim}, coords_y.ndim:{coords_y.ndim}")

    total_length:float = 0.0

    for i in range(coords_x.shape[0] - 1):
        x1, y1 = coords_x[i], coords_y[i]
        x2, y2 = coords_x[i + 1], coords_y[i + 1]

        # print(f"{int(x1), int(y1)}, {int(x2), int(y2)}")

        total_length += math.sqrt(
            math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)
        )

    return total_length


# Return lisee csa measures
def lisee_zone_average_thickness(
    lisee_zone: Polygon  
        ) -> float:
    '''
    Takes a lisee zone, and returns its average thickness as determined by the bottom line, or the bone
    -cartilage interface.
    '''
    zone_coords_x, zone_coords_y = np.array(lisee_zone.exterior.coords.xy)

    assert zone_coords_x.size > 0 and zone_coords_y.size > 0
    assert zone_coords_x.shape == zone_coords_y.shape
    assert zone_coords_x.ndim == 1 and zone_coords_y.ndim == 1

    left_x = min(zone_coords_x)
    right_x = max(zone_coords_x)

    left_indexes = []
    right_indexes = []

    for i in range(zone_coords_x.shape[0]):
        if zone_coords_x[i] == left_x:
            left_indexes.append(i)
        elif zone_coords_x[i] == right_x:
            right_indexes.append(i)
    
    if len(left_indexes) == 1:
        raise RuntimeError("lisee_zone_average_thickness polygon does not have a flat lhs")

    if len(right_indexes) == 1:
        raise RuntimeError("lisee_zone_average_thickness polygon does not have a flat rhs")

    left_index = left_indexes[0]
    max_y_left_indexes = zone_coords_y[left_index]
    for index in left_indexes:
        if zone_coords_y[index] > max_y_left_indexes:
            max_y_left_indexes = zone_coords_y[index]
            left_index = index
    
    right_index = right_indexes[0]
    max_y_right_indexes = zone_coords_y[right_index]
    for index in right_indexes:
        if zone_coords_y[index] > max_y_right_indexes:
            max_y_right_indexes = zone_coords_y[index]
            right_index = index

    index_1 = left_index
    index_2 = right_index

    if index_1 == index_2:
        raise RuntimeError("Lisee_zone_average_thickness failed to find bottom line ends.")
    
    # ensures indexes are ordered correctly when slicing coord arrays. 
    if index_1 > index_2:
        index_2, index_1 = index_1, index_2

    return lisee_zone.area / line_length(coords_x=zone_coords_x[index_1:index_2 + 1], coords_y=zone_coords_y[index_1:index_2 + 1])


# main function for FTC analysis
def FTC_analysis(
        image_array: ArrayLike, 
        roi: ImagejRoi,
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

    results_dict = {}
    # results_dict["img"] = rotated_image # return image for visualisation

    trimmed_roi_coords = trim_roi_coords_roi_based(
    roi_coords_x=rotated_roi_coords[:,0], # every column of the first row
    roi_coords_y=rotated_roi_coords[:,1],
    roi_left=left,
    roi_right=right,
    image_height=image_height,
    trim_factor=0.25
    )

    results_dict["trimmed_roi_coords"] = trimmed_roi_coords

    trimmed_roi_lobf_mid_x = roi_midpoint_lobf(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1],
        polynomial_order=4
    )

    trimmed_roi_algo_mid_x = roi_midpoint_from_algo(
        roi_coords_x=trimmed_roi_coords[0],
        roi_coords_y=trimmed_roi_coords[1]
    )

    results_dict["trimmed_roi_lobf_mid_x"] = trimmed_roi_lobf_mid_x
    results_dict["trimmed_roi_algo_mid_x"] = trimmed_roi_algo_mid_x

    lisee_polygons = lisee_CSA_polygon_function(
         image_height=image_height,
         image_width=image_width,
         roi_coords_x=trimmed_roi_coords[0],
         roi_coords_y=trimmed_roi_coords[1],
         roi_mid_x=trimmed_roi_algo_mid_x   
        )


    # TODO: save annotated image to dict, remove polygons from dict
    plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(rotated_image)
    colors = ["red", "green", "blue"]
    for i, polygon in enumerate(lisee_polygons):
            plot_polygon(polygon=polygon, ax=ax, color=colors[i])
    
    results_dict["img"] = fig
    
    results_dict["lisee_polygons"] = lisee_polygons
    results_dict["lisee_lateral_polygon"], results_dict["lisee_central_polygon"], results_dict["lisee_medial_polygon"] = lisee_polygons
    lisee_lateral_roi_polygon, lisee_central_roi_polygon, lisee_medial_roi_polygon = lisee_polygons

    results_dict["lisee_lateral_pixels"] = lisee_lateral_roi_polygon.area
    results_dict["lisee_central_pixels"] = lisee_central_roi_polygon.area
    results_dict["lisee_medial_pixels"] = lisee_medial_roi_polygon.area

    results_dict["lisee_lateral_area_mm"] = pixels_to_mm(lisee_lateral_roi_polygon.area)
    results_dict["lisee_central_area_mm"] = pixels_to_mm(lisee_central_roi_polygon.area)
    results_dict["lisee_medial_area_mm"] = pixels_to_mm(lisee_medial_roi_polygon.area)

    lisee_lateral_average_thickness_pixles = lisee_zone_average_thickness(lisee_zone=lisee_lateral_roi_polygon)
    lisee_central_average_thickness_pixles = lisee_zone_average_thickness(lisee_zone=lisee_central_roi_polygon)
    lisee_medial_average_thickness_pixles = lisee_zone_average_thickness(lisee_zone=lisee_medial_roi_polygon)

    results_dict["lisee_lateral_average_thickness_pixles"] = pixels_to_mm(lisee_lateral_average_thickness_pixles)
    results_dict["lisee_central_average_thickness_pixles"] = pixels_to_mm(lisee_central_average_thickness_pixles)
    results_dict["lisee_medial_average_thickness_pixles"]  = pixels_to_mm(lisee_medial_average_thickness_pixles )

    results_dict["lisee_lateral_average_thickness_mm"] = lisee_lateral_average_thickness_pixles
    results_dict["lisee_central_average_thickness_mm"] = lisee_central_average_thickness_pixles
    results_dict["lisee_medial_average_thickness_mm"]  = lisee_medial_average_thickness_pixles 

    # TODO: add echo intensity
    rotated_image_array = np.array(rotated_image)
    third: float = 1.0 / 3.0

    # rotated_image_array_gryscl = np.dot(rotated_image_array[..., :3], [third,third,third]) # remove alpha, dot product rgb channels

    # rotated_image_array_gryscl_width: int = rotated_image_array_gryscl.shape[0]
    # rotated_image_array_gryscl_height: int = rotated_image_array_gryscl.shape[1]


    # results_dict["lisee_lateral_ei"] = average_ei(
    #     image_width=rotated_image_array_gryscl_width,
    #     image_height=rotated_image_array_gryscl_height,
    #     polygon=lisee_lateral_roi_polygon,
    #     image_array=rotated_image_array_gryscl)

    # results_dict["lisee_central_ei"] = average_ei(
    #     image_width=rotated_image_array_gryscl_width,
    #     image_height=rotated_image_array_gryscl_height,
    #     polygon=lisee_central_roi_polygon,
    #     image_array=rotated_image_array_gryscl)

    # results_dict["lisee_medial_ei"] = average_ei(
    #     image_width=rotated_image_array_gryscl_width,
    #     image_height=rotated_image_array_gryscl_height,
    #     polygon=lisee_medial_roi_polygon,
    #     image_array=rotated_image_array_gryscl)


    # pnt1 = Point(500, 183)
    # print(lisee_lateral_roi_polygon.exterior.coords.xy)
    # print(contains_xy(lisee_lateral_roi_polygon, 183, 500))

    # fig, ax = plt.subplots()

    # ax.imshow(rotated_image)
    # plot_points(pnt1, ax=ax)

    # # for i, polygon in enumerate(lisee_polygons):
    # #     plot_polygon(polygon=polygon, ax=ax)

    # plot_polygon(polygon=lisee_lateral_roi_polygon, ax=ax)


    # pyplot.show()

    return results_dict
    

    # Calcualte Lisee EIs

    # Calcualte Hori thicknesses

    # Calculate Hori CSA



'''
For testing only
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
