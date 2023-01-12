import numpy as np
from scipy import sparse as sp
from scipy.sparse import coo_matrix

from gns.utils.normalized_adjacency import normalized_adjacency


def normalized_laplacian(A, symmetric=True) -> coo_matrix:
    """
    Calculates the normalized Laplacian of a given adjacency matrix.

    Args:
        A: rank 2 array or sparse matrix
        symmetric: bool, a sign of whether to calculate symmetric nomalization

    Returns:
        normalized laplacian
    """
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)

    normalized_adj = normalized_adjacency(A, symmetric=symmetric)

    return I - normalized_adj
