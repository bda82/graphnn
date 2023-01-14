from spektral.utils.sparse import sp_matrix_to_sp_tensor


class AdjToSpTensor:
    """
    Converts the adjacency matrix to a SparseTensor.
    """

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = sp_matrix_to_sp_tensor(graph.a)

        return graph
