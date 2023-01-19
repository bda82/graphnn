from gns.utils.normalized_adjacency_matrix import normalized_adjacency_matrix


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
            graph.a = normalized_adjacency_matrix(graph.a, self.symmetric)

        return graph
