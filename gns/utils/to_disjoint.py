import numpy as np
from scipy import sparse as sp


def to_disjoint(x_list=None, a_list=None, e_list=None) -> tuple:
    """
    Converts lists of node objects, adjacency matrices, and boundary objects into disjoint mode.

    Either node objects or adjacency matrices must be provided as input.

    the i-th element of each list must be associated with the i-th graph.

    The method also computes a batch index to extract individual graphs from a disjoint set
    and they will always be returned as a stacked list of edges.

    Edge attributes can be represented as:
        - a dense array of the form `(n_nodes, n_nodes, n_edge_features)`
        - list of sparse edges of the shape `(n_edges, n_edge_features)`

    Note:
    The n_nodes parameter can change from graph to graph;

    Args:
        x_list: list of Numpy arrays `np.arrays` of the form `(n_nodes, n_node_features)`
        a_list: list of Numpy arrays `np.arrays` or matrices `scipy.sparse` of the form `(n_nodes, n_nodes)`
        e_list: list of Numpy arrays `np.arrays` of the form
                `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`
    
    Returns:
        x: np.array of shape (n_nodes, n_node_features)
        a: scipy.sparse matrix of shape (n_nodes, n_nodes)
        e: np.array of the form (n_edges, n_edge_features);
        i: np.array of the form (n_nodes, );
    """
    if a_list is None and x_list is None:
        raise ValueError("Need at least x_list or a_list.")

    # Node features

    x_out = None
    if x_list is not None:
        x_out = np.vstack(x_list)

    # Adjacency matrix

    a_out = None
    if a_list is not None:
        a_out = sp.block_diag(a_list)

    # batch index

    n_nodes = np.array([x.shape[0] for x in (x_list if x_list is not None else a_list)])
    i_out = np.repeat(np.arange(len(n_nodes)), n_nodes)

    # edges attribures

    e_out = None
    if e_list is not None:
        if e_list[0].ndim == 3:
            # Here you need to convert a dense matrix format to a sparse one
            e_list = [e[sp.find(a)[:-1]] for e, a in zip(e_list, a_list)]
            
        e_out = np.vstack(e_list)

    return tuple(out for out in [x_out, a_out, e_out, i_out] if out is not None)
