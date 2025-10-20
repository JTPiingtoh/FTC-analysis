import numpy as np
from shapely import Polygon, Point


def get_average_ei(
    image_width: int,
    image_height: int,
    polygon: Polygon,
    image_array: np.array,
    
    ):
    
    average_ei = 0
    n = 0

    x_coords, y_coords = polygon.exterior.coords.xy

    # print(x_coords, y_coords)

    left = int(np.min(x_coords))
    right = int(np.max(x_coords))
    top = int(np.min(y_coords))
    bottom = int(np.max(y_coords))

    # iterate through points of the image
    for x in range(left, right - 1):
        for y in range(top, bottom - 1):
            
            if not polygon.contains(Point(x, y)):
                continue
            
            # x and y are flipped for imagej rois 
            average_ei = ( (average_ei * n) + image_array[y][x] ) / (n + 1)
            n += 1


    return average_ei




'''
For testing only
'''
if __name__ == "__main__":

    from tifffile import TiffFile
    from roifile import ImagejRoi
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon

    from roi_rotate import rotate_image_and_roi
    from roi_trim import trim_roi_coords_roi_based
    from PIL import Image
    from midpoint_lobf import roi_midpoint_lobf
    from roi_rotate import rotate_image_and_roi
    from roi_trim import trim_roi_coords_roi_based
    from lisee import lisee_CSA_polygon_function, lisee_zone_average_thickness

    with TiffFile('14_annotated.tif') as tif:
        
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
        # ax.plot(left, top, 'go')
        # ax.plot(right, bottom, 'go')
  
        # plot_polygon(trimmed_roi_polygon, color="red", ax=ax)        
        # ax.axvline(results["trimmed_roi_lobf_mid_x"])

        colors = ["red", "green", "blue" ,"purple"]

        rotated_image_array = np.array(rotated_image)
        third: float = 1.0 / 3.0
        rotated_image_array_gryscl = np.dot(rotated_image_array[..., :3], [third,third,third]) # remove alpha, dot product rgb channels


        for i, polygon in enumerate(lisee_polygons):
            plot_polygon(polygon=polygon, ax=ax, color=colors[i])
            print(get_average_ei(
                image_width=image_width,
                image_height=image_height,
                polygon=polygon,
                image_array=rotated_image_array_gryscl
                )
            )


        plt.show()    