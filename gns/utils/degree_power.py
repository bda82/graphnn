import warnings
from typing import Any

import numpy as np
from scipy import sparse as sp


def calculate_degree_power(A, k) -> Any:
    """
    Calculates the value of `A` in power of `k` from the given adjacency matrix.
    It can be used to calculate the normalized Laplacian.

    Args:
        A: rank 2 array or sparse matrix
        k: exponent to which the degree matrix is raised
        
    Returns:
        If the input parameter `A` is a dense array, it will return a dense array;
        If `A` is sparse, it will return a sparse array in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()

    degrees[np.isinf(degrees)] = 0.0

    issparse_predicate = sp.issparse(A)

    if issparse_predicate:
        diags = sp.diags(degrees)
    else:
        diags = np.diag(degrees)

    return diags
