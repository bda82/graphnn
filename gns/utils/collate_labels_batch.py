import numpy as np

from gns.utils.pad_jagged_array import pad_jagged_array


def collate_labels_batch(y_list, node_level=False):
    """
    Collate labels.
    
    Args:
    
    Returns:
    
    """
    if node_level:
        n_max = max([x.shape[0] for x in y_list])
        return pad_jagged_array(y_list, (n_max, -1))
    else:
        return np.array(y_list)
