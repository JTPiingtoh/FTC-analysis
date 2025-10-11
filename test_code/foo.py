import numpy as np
from numpy.polynomial.polynomial import polyvalfromroots

x = np.array([1,2,3])
y = np.array([4,5,6])
xy = np.vstack([x,y]).T

print(xy)