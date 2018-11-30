import math
import sys

import numba
from numba import cuda
import numpy as np
import time

start = time.time()

vran = 100

if len(sys.argv)>1:
    loop = int(sys.argv[1])
else:
    loop = 1000

if len(sys.argv) > 2:
    vlen = int(sys.argv[2])
else:
    vlen = 65536

print ('running with %s loops and vectorlength %s' % (loop,vlen))

pm = np.empty((vran,vran))
for i in range(vran):
    for j in range(vran):
        pm[i,j]=i*j

@cuda.jit
def inner(dv1,dv2,dvresult):
    pos = cuda.grid(1)
    denom_a = 0
    denom_b = 0
    dot = 0
    v1 = dv1
    v2 = dv2

    if pos < loop:
        for i in range(v1.size):
            a = v1[i]
            b = v2[i]

            denom_a += a*a
            denom_b += b*b
            dot += a * b

        dvresult[pos]= pos+(dot /math.sqrt(float(denom_a)*float(denom_b)))

        #cosin = pos+(dot /math.sqrt(float(denom_a)*float(denom_b)))

@cuda.jit
def lastelement(out):
    out[0]= 1

@cuda.reduce
def sum_reduce(a, b):
    return a + b

stream = cuda.stream()



v1 = np.random.randint(0,vran,size=vlen,dtype=np.uint32)
v2 = np.random.randint(0,vran,size=vlen,dtype=np.uint32)
dv1 = cuda.to_device(v1)
dv2 = cuda.to_device(v2)


vresults = np.zeros(loop, dtype=np.float)
dvresults = cuda.to_device(vresults)

threadsperblock = 32
blockspergrid = (loop + (threadsperblock - 1)) // threadsperblock

out = cuda.to_device(np.zeros(1))
inner[blockspergrid, threadsperblock](dv1,dv2,dvresults) #der eigentliche GPU Aufruf
print(dvresults.copy_to_host()[-1])
#print(out.copy_to_host( ))
#print(out[0])
print('total',time.time()-start)




