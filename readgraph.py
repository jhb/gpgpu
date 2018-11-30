import os

def readgraph(size):
    base = os.getcwd()

    nodesfile = open(os.path.join(base,'nodes_%s.txt' % size),'r')
    nodes = []
    offset = 0
    for line in nodesfile.readlines():
        line = line.strip()
        nid,num_edges = map(int,line.split())
        nodes.append((offset,num_edges))
        offset += num_edges

    edgesfile = open(os.path.join(base, 'edges_%s.txt' % size), 'r')
    edges = []
    for line in edgesfile.readlines():
        line = line.strip()
        source, target = map(int, line.split())
        edges.append((source,target))

    return (nodes,edges)

if __name__=='__main__':
    print(readgraph(1000))