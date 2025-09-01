import numpy as np

flat_pairs = [((1,10),(3,10)),
              ((6,5),(2,5))]

flat_pairs = np.array(flat_pairs)
max = np.max(flat_pairs[:,0,1])

x = np.mean(flat_pairs[flat_pairs[:,0,1] == np.max(flat_pairs[:,0,1])][0].T[0])

print(x)
print(max)