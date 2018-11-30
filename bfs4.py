"""
(c) Joerg Baach 2018, GPLv3

"""

import readgraph
import time
import numpy as np
import numba
from numba import njit,prange,jit
from numba import cuda

nodes,edges = readgraph.readgraph(1000000)
print('finished reading')
start = time.time()


nodearray = np.array(nodes, dtype=np.uint32)
dnodearray = cuda.to_device(nodearray)

edgearray = np.array(edges, dtype=np.uint32)
dedgearray = cuda.to_device(edgearray)

unknown = np.ones(len(nodes))

frontier = set()
result = []


s = 1
frontier.add(s)
unknown[s]=0
dunknown = cuda.to_device(unknown)


@jit(nopython=True)
def buildEdgelist(frontier, nodearray):
    edgelist = []
    for nodeid in frontier:
        offset, num = nodearray[nodeid]
        for n in range(num):
            edgelist.append(offset+n)
    return edgelist

@cuda.jit
def workEdges(edgelist,edgearray,unknown,targetlist):
    pos = cuda.grid(1)
    if pos<edgelist.size:
        source,target = edgearray[edgelist[pos]]
        targetlist[pos]=unknown[target]*target
        unknown[target] = 0


def advance(nodearray,edgearray,unknown,frontier):
    edgelist = buildEdgelist(frontier, nodearray)  #host?
    
    targetlist = np.empty(len(edgelist),dtype=np.uint32)
    dedgelist = cuda.to_device(np.array(edgelist,dtype=np.uint32))
    dtargetlist = cuda.to_device(targetlist)
    
    threadsperblock = 32
    blockspergrid = (len(edgelist) + (threadsperblock - 1)) // threadsperblock
    workEdges[blockspergrid, threadsperblock](dedgelist,dedgearray,dunknown,dtargetlist)

    targetlist = dtargetlist.copy_to_host()
    newfrontier = set(targetlist) #host
    newfrontier.discard(0) #host
    return newfrontier

result = []
maxfrontier = 0
while frontier:
    maxfrontier = max(maxfrontier,len(frontier))
    result.extend(frontier)
    frontier = advance(nodearray, edgearray, unknown, frontier)
    print(len(result),len(frontier))

print()
print(maxfrontier)
print(len(result))
print(time.time()-start)

# 1m   1.8845374584197998
# 10m 14.704204320907593
