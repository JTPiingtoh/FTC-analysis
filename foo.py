import numpy as np

arr = np.array([2,3,5,6,1,2,4,5])
arr_indexes = np.arange(arr.shape[0])

empty = np.array([])

indexes = arr_indexes[arr == 5]

assert empty.size > 0
