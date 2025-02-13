'''
    Code adapted from https://github.com/cgohlke/roifile/tree/master by Christoph Gohlke
''' 

import numpy
from matplotlib import pyplot
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box,intersection
from shapely.plotting import plot_line, plot_polygon


def plot_image_overlays(image, overlays, **kwargs):
    """Plot image and overlays (bytes) using matplotlib."""
    fig, ax = pyplot.subplots()
    ax.imshow(image, cmap='gray')
    if not isinstance(overlays, list):
        overlays = [overlays]
    for overlay in overlays:
        roi = ImagejRoi.frombytes(overlay)
        roi.plot(ax, **kwargs)
    pyplot.show()


with TiffFile('502 with roi.tif') as tif:
    try:
        image = tif.pages[0].asarray()
        assert tif.imagej_metadata is not None
        overlays = tif.imagej_metadata['ROI']
        roi = ImagejRoi.frombytes(overlays)
        coords = roi.integer_coordinates
        left = roi.left
        top = roi.top
        coords += [left, top]

        polygon = Polygon(coords)
        
        print(f" area: {polygon.area}")
        width = roi.widthd
        print(f"left: {roi.left}")
        print(f"right: {roi.right}")


        fig ,ax = pyplot.subplots()
        ax.imshow(image)
        ax.plot(coords[:, 0], coords[:, 1])
        # plot box
        height = image.shape[0]
        width = image.shape[1]
        my_box = box(100, 0, width / 2, height )
        plot_line(my_box.boundary, ax=ax,)
        
        # Plot intersection
        intersect = intersection(polygon, my_box)
        plot_polygon(intersect, ax=ax, color="red")
        print(f"Area of intersection: {intersect.area}")


        pyplot.show()

    except AssertionError:
        print("Image has no metadata") # TODO: Change to image name for main function

# plot_image_overlays(image, overlays)

