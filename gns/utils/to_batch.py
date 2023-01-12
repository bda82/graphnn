import numpy as np

from gns.utils.pad_jagged_array import pad_jagged_array


def to_batch(x_list=None, a_list=None, e_list=None, mask=False):
    """
    Converts lists of node features, adjacency matrices and edge features to
    `[batch mode]`, by zero-padding all tensors to have the same node dimension `n_max`.

    Either the node features or the adjacency matrices must be provided as input.

    The i-th element of each list must be associated with the i-th graph.

    If `a_list` contains sparse matrices, they will be converted to dense
    np.arrays.

    The edge attributes of a graph can be represented as
        - a dense array of shape `(n_nodes, n_nodes, n_edge_features)`;
        - a sparse edge list of shape `(n_edges, n_edge_features)`;

    They will always be returned as dense arrays.

    Args:
        x_list: a list of np.arrays of shape `(n_nodes, n_node_features)` - note that `n_nodes` can change between graphs;
        a_list: a list of np.arrays or scipy.sparse matrices of shape `(n_nodes, n_nodes)`;
        e_list: a list of np.arrays of shape `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`;
        mask: bool, if True, node attributes will be extended with a binary mask that
                    indicates valid nodes (the last feature of each node will be 1 if the node is valid
                    and 0 otherwise). Use this flag in conjunction with layers.base.GraphMasking to 
                    start the propagation of masks in a model.

    Returns:
        only if the corresponding list is given as input:
        `x`: np.array of shape `(batch, n_max, n_node_features)`;
        `a`: np.array of shape `(batch, n_max, n_max)`;
        `e`: np.array of shape `(batch, n_max, n_max, n_edge_features)`;
    """
    if a_list is None and x_list is None:
        raise ValueError("Need at least x_list or a_list")

    n_max = max([x.shape[0] for x in (x_list if x_list is not None else a_list)])

    # Node features

    x_out = None
    
    if x_list is not None:
        if mask:
            x_list = [np.concatenate((x, np.ones((x.shape[0], 1))), -1) for x in x_list]
        x_out = pad_jagged_array(x_list, (n_max, -1))

    # Adjacency matrix
    
    a_out = None
    
    if a_list is not None:
        if hasattr(a_list[0], "toarray"):  # Convert sparse to dense
            a_list = [a.toarray() for a in a_list]
        a_out = pad_jagged_array(a_list, (n_max, n_max))

    # Edge attributes
    
    e_out = None
    
    if e_list is not None:
        if e_list[0].ndim == 2:  # Sparse to dense
            for i in range(len(a_list)):
                a, e = a_list[i], e_list[i]
                e_new = np.zeros(a.shape + e.shape[-1:])
                e_new[np.nonzero(a)] = e
                e_list[i] = e_new
        e_out = pad_jagged_array(e_list, (n_max, n_max, -1))

    return tuple(out for out in [x_out, a_out, e_out] if out is not None)
