import numpy as np

from gns.utils.generate_jagged_array_for_arbitrary_dimensions import generate_jagged_array_for_arbitrary_dimensions


def collate_labels_batch(y_list, node_level=False):
    """
    Collate labels.
    
    Args:
    
    Returns:
    
    """
    if node_level:
        n_max = max([x.shape[0] for x in y_list])
        return generate_jagged_array_for_arbitrary_dimensions(y_list, (n_max, -1))
    else:
        return np.array(y_list)
