import numpy as np
from numpy.polynomial.polynomial import polyvalfromroots

arr = np.array([[1,2],
               [2,4]]).flatten()

x = polyvalfromroots(4, arr)

print(x)