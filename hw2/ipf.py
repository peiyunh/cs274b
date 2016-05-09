import numpy as np 
D = np.genfromtxt('data/data.txt', delimiter=None, dtype='int')
E = np.genfromtxt('data/edges.txt', dtype='int')
loc = np.genfromtxt('data/locations.txt', delimiter=None)
m,n = D.shape  # m = 2760 data points, n=30 dimentional

# Problem 2 (a)
from graphviz import Digraph
G = Digraph(comment="Loopy")
for i in xrange(n):
    lbl = 'station #%d\n(lat=%.4f, lon=%.4f)'%(i, loc[i][0],loc[i][1])
    G.node(str(i),label=lbl)
for (xi,xj) in E:
    G.edge(str(xi),str(xj))
G.render('output/loopy.png', view=True)

#
import pyGM as gm
X = [gm.Var(i,2) for i in xrange(n)]
fh = list()
for (i,j) in E:
    fh.append(gm.Factor([X[i],X[j]],0.0))

for s in xrange(len(D)):
    for ij,fij in enumerate(fh):
        xij = fij.vars
        fh[ij][ D[s,xij[0]], D[s,xij[1]] ] += 1.0

### Problem 2 (b) 
ph = fh
for ij in xrange(len(ph)):
    ph[ij] /= ph[ij].sum()
    #print '(%d,%d): ' %(ph[ij].vars[0], ph[ij].vars[1])
    #print ph[ij].table


### Problem 2(c)
f = []
fmap = {}
for (i,j) in E:
    fmap[(i,j)] = len(f) 
    f.append(gm.Factor([X[i],X[j]], 1.0))

import ipdb
niter = 10
# for each iteration
for it in xrange(niter):
    # for each edge
    for (i,j) in E:
        # variational elimination for marginal probability 
        model = gm.GraphModel(f)
        
        pri = [1.0 for Xi in X]
        pri[i], pri[j] = 2.0, 2.0
        order = gm.eliminationOrder(model, orderMethod='minfill', priority=pri)[0]

        sumElim = lambda F, Xlist: F.sum(Xlist)
        model.eliminate(order[:-2], sumElim)

        pij = model.joint()
        lnZ = np.log(pij.sum())
        #print 'lnZ: ', lnZ
        pij /= pij.sum()
        #print pij, '\n', pij.table

        #
        ij = fmap[(i,j)]
        f[ij] *= ph[ij] / pij

    print 'iter %d' % it
#
