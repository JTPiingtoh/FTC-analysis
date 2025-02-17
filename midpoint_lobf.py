import numpy as np
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box,intersection, LineString
from shapely.plotting import plot_line, plot_polygon
from typing import Literal



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
        fitted = np.polynomial.polynomial.polyfit(coords[:,0], coords[:,1], 3)
        print(fitted)

        fig ,ax = plt.subplots()      
        ax.imshow(image)
        plot_polygon(polygon=polygon, ax=ax)
        ax.plot(fitted)

        plt.show()