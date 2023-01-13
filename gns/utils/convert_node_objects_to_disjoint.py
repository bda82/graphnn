import numpy as np
from scipy import sparse as sp


def convert_node_objects_to_disjoint(x_list=None, a_list=None, e_list=None) -> tuple:
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
        x: `np.array` of shape `(n_nodes, n_node_features)`
        a: `scipy.sparse` matrix of shape `(n_nodes, n_nodes)`
        e: `np.array` of the form `(n_edges, n_edge_features)`
        i: `np.array` of the form `(n_nodes, )`
    """
    if a_list is None and x_list is None:
        raise ValueError("Should be at list one parameter: x_list or a_list.")

    # Node features

    node_features_output = None

    if x_list is not None:
        node_features_output = np.vstack(x_list)

    # Adjacency matrix

    adjacency_matrix_output = None

    if a_list is not None:
        adjacency_matrix_output = sp.block_diag(a_list)

    # Batch indexes

    number_of_nodes = np.array([x.shape[0] for x in (x_list if x_list is not None else a_list)])

    index_output = np.repeat(np.arange(len(number_of_nodes)), number_of_nodes)

    # Edge features

    edge_features_output = None

    if e_list is not None:
        if e_list[0].ndim == 3:
            # Convert sparse to dense
            e_list = [e[sp.find(a)[:-1]] for e, a in zip(e_list, a_list)]
            
        edge_features_output = np.vstack(e_list)

    return tuple(
        out
        for out in [node_features_output, adjacency_matrix_output, edge_features_output, index_output]
        if out is not None
    )
