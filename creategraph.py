import os
import random
import sys

n = 10000000

nodes = []
edges = {}

for i in range(n*2):
    while True:
          start = random.randint(1,n)
          target = random.randint(1,n)
          targets = edges.setdefault(start,[])
          if target not in targets:
              targets.append(target)
              break

edgelist = []
offset = 0

for i in range(n+1):
    targets = edges.setdefault(i,[])
    for t in sorted(targets):
        edgelist.append((i,t))
    nodes.append((offset,len(targets)))
    offset+=len(targets)

base = os.getcwd()

nodesfile = open(os.path.join(base,'nodes_%s.txt' % n),'w')
lb = ''
for node in nodes:
    nodesfile.write('%s%s %s' % (lb,node[0],node[1]))
    lb = '\n'

edgesfile = open(os.path.join(base,'edges_%s.txt' % n),'w')
lb = ''
for e in edgelist:
    edgesfile.write('%s%s %s' % (lb,e[0],e[1]))
    lb = '\n'