import numpy as np
from numpy.polynomial import Polynomial, polynomial 
import enum
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from roifile import ImagejRoi
from tifffile import TiffFile
from shapely import Polygon, box,intersection, LineString
from shapely.plotting import plot_line, plot_polygon
from typing import Literal
from PIL import Image

from roi_rotate import rotate_image_and_roi


def closest_to(
    x: np.ndarray,
    n: int|float
) -> int:
    ''' Returns numbers in x closest to n by index, with x being a 1-dimensional array'''
    assert x.ndim == 1

    differences_array = np.array(abs(n-x))
    return x[differences_array == min(differences_array)]


def roi_midpoint_lobf(
    roi_coords_x: np.ndarray,
    roi_coords_y: np.ndarray,
    polynomial_order: int = 4,
) -> float:
    '''As  FTC ROIs tend to have a recurve shape, their overall geometry can be approximated by plotting 
        a line of best fit using an nth order polynomial. Calculating the point at which the gradient
        of this polynomial is zero close to the half-width of the roi gives a good approximation of the
        roi midpoint.
    '''
    assert roi_coords_x.ndim == 1
    assert roi_coords_y.ndim == 1
    
    roi_width = max(roi_coords_x) - min(roi_coords_x)
    roi_left = float(min(roi_coords_x))

    if roi_width <= 0:
        raise ValueError("Roi has zero width")

    # Line of best fit using mean squares
    coefs = polynomial.polyfit(roi_coords_x, roi_coords_y, polynomial_order)
    roi_polynomial = Polynomial(coefs) # TODO: this is redundant 

    # calculate derivative and solve for 0
    roi_polynomial_der1_coefs = polynomial.polyder(roi_polynomial.coef)
    lobf_roots = polynomial.polyroots(roi_polynomial_der1_coefs)
    real_roots = lobf_roots[~np.iscomplex(lobf_roots)].real # ~ bitwise not
    positive_roots = real_roots[real_roots > 0] # complex and negative real numbers are disgarded 

    return closest_to(x=positive_roots, n=(roi_width/2) + roi_left) # numbers closest to half-width is calculated


if __name__ == "__main__":
    with TiffFile('14_annotated_with_border.tif') as tif:
        
        image = tif.pages[0].asarray()
        image = Image.fromarray(image)
        assert tif.imagej_metadata is not None
        overlays = tif.imagej_metadata['ROI']
        roi = ImagejRoi.frombytes(overlays)
        coords = roi.integer_coordinates
        left = roi.left
        top = roi.top      
        coords += [left, top]
        

        # height = image.shape[0]
        # width = image.shape[1]
        # line = LineString([[width / 1.2 , 0], [width / 1.2, height]])
        # coefs = polynomial.polyfit(coords[:,0], coords[:,1], 7)
        # roi_polynomial = Polynomial(coefs)
        
        # roi_polynomial_der1_coefs = polynomial.polyder(roi_polynomial.coef)
        
        # roi_polynomial_der1 = Polynomial(roi_polynomial_der1_coefs)

        
        # xs = np.arange(0, width, 1)

        # lobf = roi_polynomial(xs)
        # lobf_der1 = roi_polynomial_der1(xs)
        # lobf_roots = polynomial.polyroots(roi_polynomial_der1_coefs)
        # real_roots = lobf_roots[~np.iscomplex(lobf_roots)].real
        # positive_roots = real_roots[real_roots > 0]
        # mid = closest_to(x=positive_roots, n=width/2 + left)[0]
     
        image, coords = rotate_image_and_roi(image, roi)
        polygon = Polygon(coords)

        mid = roi_midpoint_lobf(coords[:,0], coords[:,1])
        fig ,ax = plt.subplots()      
        ax.imshow(image)

        # ax.plot(lobf_der1*100, color="green")
        ax.axvline(mid, color="green")
        plot_polygon(polygon=polygon, ax=ax)
        

        plt.show()