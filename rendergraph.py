import os

import readgraph
from graphviz import Digraph

dot = Digraph()
nodes, edges = readgraph.readgraph(10)

for i in range(1,len(nodes)+1):
    dot.node(str(i))

for edge in edges:
    dot.edge(str(edge[0]),str(edge[1]))

dot.render(os.path.join(os.getcwd(),'graph'),format='png')

