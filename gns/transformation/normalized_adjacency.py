from gns.utils.normalized_adjacency import normalized_adjacency


class NormalizeAdjacencyMatrix:
    """
    Normalizes the adjacency matrix.

    Args:
        symmetric: bool a feature indicating whether to calculate symmetric normalization
    """

    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = normalized_adjacency(graph.a, self.symmetric)

        return
