def read_knn_graph(filename):
    """Read kNN graph from file"""
    graph = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ':' not in line:
                continue
            node, neighbors = line.split(":", 1)
            node = int(node.strip())
            neighbors = [int(n.strip()) for n in neighbors.strip().split()]
            graph[node] = neighbors
    return graph
