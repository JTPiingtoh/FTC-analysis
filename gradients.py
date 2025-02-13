import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

arr = np.array([[1, 3, 5, 8, 9, 12], # x
               [5, 3, 5, 3, 5, 3]]) # y

gradients = []
signs = []
for i in range(arr.shape[1]):
    x1 = arr[0][i]
    y1 = arr[1][i]

    if i == arr.shape[1] - 1:
        x2 = arr[0][0]
        y2 = arr[1][0]

    else:
        x2 = arr[0][i+1]
        y2 = arr[1][i+1]
    
    gradient = ((y2 - y1) / (x2 - x1))
    gradients.append(gradient)

    if gradient > 0:
        signs.append(1)
    elif gradient < 0:
        signs.append(-1)
    else:
        signs.append(0)
    
negative_to_positive_list = []
diffs = []
for j in range(len(signs)):

    diff = signs[j] - signs[j-1]
    if diff > 0:
        negative_to_positive_list.append(True)
    elif diff < 0:
        negative_to_positive_list.append(False)

    diffs.append(diff)

infls = np.where(diffs)

print(gradients)
print(signs)
print(diffs)
print(infls[0])
print(negative_to_positive_list)
plt.plot(arr[0], arr[1])
plt.show()