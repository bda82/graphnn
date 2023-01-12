from typing import Any

import numpy as np
from numpy import ndarray


def collate_labels_disjoint(y_list, node_level=False) -> ndarray | Any:
    """
    Matches this list of labels for the disjoint (disjoint) mode.

    Notes:
        If the `node_level` parameter equivals False, the labels can be scalars or should have the form z[n_labels, ]z.

    If the node_level parameter=True, the labels must be in the form of z[n_nodes, ]z (i.e. a scalar label for each node)
    or z[n_nodes, n_labels]z.

    Args:
        y_list: list of Numpy arrays np.arrays or scalars.
        node_level: bool parameter, indicates whether labels should be packed both at the node level or at the graph level.
    
    Returns:
        If the node_level parameter=False: returns np.array of the form [len(y_list), n_labels].
        If the node_level parameter=True: returns np.array of the form [n_nodes_total, n_labels], where
        the parameter n_nodes_total = sum(y.form[0] for y in y_list).
    """
    if node_level:
        if len(np.shape(y_list[0])) == 1:
            y_list = [y_[:, None] for y_ in y_list]

        return np.vstack(y_list)
    else:
        if len(np.shape(y_list[0])) == 0:
            y_list = [np.array([y_]) for y_ in y_list]

        return np.array(y_list)
