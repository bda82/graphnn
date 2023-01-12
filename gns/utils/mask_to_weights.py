import numpy as np


def mask_to_simple_weights(mask) -> int:
    """
    Converts the bitmask to simple weights to calculate the average losses across the network nodes.
    
    Args:
        mask: mask
    
    Returns: 
        mask
    """
    return mask / np.count_nonzero(mask)
