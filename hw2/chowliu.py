### Problem 1 (a) 
# Load the data points D, and the station locations (lat/lon)
import numpy as np
D = np.genfromtxt('data/data.txt', delimiter=None, dtype='int')
loc = np.genfromtxt('data/locations.txt', delimiter=None)
m,n = D.shape  # m = 2760 data points, n=30 dimentional
# D[i,j] = 1 if station j observed rainfall on day i 


### Problem 1 (b)
# Compute emprirical mutual information score 
import pyGM as gm
X = [gm.Var(i,2) for i in xrange(n)] # X has n vars and each has 2 outcomes 

unifactors = list()
pairfactors = list()
for i in xrange(n):
    unifactors.append(gm.Factor([X[i]],0.0))
    for j in xrange(i+1,n):
        pairfactors.append(gm.Factor([X[i],X[j]],0.0))

# load data into factors 
for s in xrange(len(D)):
    for i, fi in enumerate(unifactors):
        xi = fi.vars
        unifactors[i][ D[s,xi[0]], ] += 1.0
    for ij,fij in enumerate(pairfactors):
        xij = fij.vars
        pairfactors[ij][ D[s,xij[0]], D[s,xij[1]] ] += 1.0

# normalize to joint marginal probabilities 
uniprobs = unifactors
for i, fi in enumerate(uniprobs):
    uniprobs[i] /= uniprobs[i].sum()
        
pairprobs = pairfactors
for ij, fij in enumerate(pairprobs):
    pairprobs[ij] /= pairprobs[ij].sum()

MI = np.zeros((n,n)) 
for ij, pij in enumerate(pairprobs):
    xi = pij.vars[0]
    xj = pij.vars[1]
    mi = uniprobs[xi].entropy() + uniprobs[xj].entropy() - pairprobs[ij].entropy()
    MI[xi][xj] = mi
    MI[xj][xi] = mi

### Problem 1 (c)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
W = csr_matrix(-MI)
T = minimum_spanning_tree(W)
rows, cols = T.nonzero()
print 'Edges: ', zip(rows, cols)

from graphviz import Digraph
G = Digraph(comment="MST")

for i in xrange(n):
    lbl = 'station #%d\n(lat=%.4f, lon=%.4f)'%(i, loc[i][0],loc[i][1])
    G.node(str(i),label=lbl)

for (xi,xj) in zip(rows,cols):
    w = MI[xi][xj]
    G.edge(str(xi),str(xj),label=str(w))

G.render('output/MST', view=True)

### Problem 1(d)
# report the average log-likelihood
P = gm.Factor([],1.0)
for ij, pij in enumerate(pairprobs):
    xi, xj = pij.vars[0].label, pij.vars[1].label
    if T[xi,xj] != 0 or T[xj,xi] != 0:
        P = P * pairprobs[ij]

LL = np.mean([np.log(P[xj]) for xj in D])
print 'Average log-likelihood:', LL
