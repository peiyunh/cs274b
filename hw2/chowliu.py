import numpy as np
# Load the data points D, and the station locations (lat/lon)
D = np.genfromtxt('data/data.txt', delimiter=None)
loc = np.genfromtxt('data/locations.txt', delimiter=None)
m,n = D.shape  # m = 2760 data points, n=30 dimentional
# D[i,j] = 1 if station j observed rainfall on day i 

import pyGM as gm
X = [gm.Var(i,2) for i in range(n)]
p01 = gm.Factor([X[0], X[1]], 0.0)

