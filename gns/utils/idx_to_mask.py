import numpy as np
from numpy import ndarray


def idx_to_mask(idx, l) -> ndarray:
    """
    Returns mask by indexes.

    Args:
        idx: index
        l: dimension
    
    Returns: 
        mask
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
