# gpgpu
Experiments with GPGPU using python

Here you find example code from me learning to use CUDA for Graphs, using python and numba.

- [python_cuda_graph.md](python_cuda_graph.md) - the main article
- [creategraph.py](creategraph.py) creates a random sample graph, and writes it out to the files nodes_10.txt and
  edges_10.txt (for a graph with 10 nodes)
- [rendergraph.py](rendergraph.py) visualizes the graph using graphviz/dot, e.g. into
  [graph.png](graph.png)
- [readgraph.py](readgraph.py) is used to read in the data
- [bfs4.py](bfs4.py) is an example to run a breadth-first-search on the graph, using CUDA.