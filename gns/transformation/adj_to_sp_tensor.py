from gns.utils.sparse_matrix_to_sparse_tensor import sparse_matrix_to_sparse_tensor


class AdjToSpTensor:
    """
    Converts the adjacency matrix to a SparseTensor.
    """

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = sparse_matrix_to_sparse_tensor(graph.a)

        return graph
