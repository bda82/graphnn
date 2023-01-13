import numpy as np
from scipy import sparse as sp
from scipy.sparse import coo_matrix

from gns.utils.normalized_adjacency_matrix import normalized_adjacency_matrix


def normalized_laplacian(A, symmetric=True) -> coo_matrix:
    """
    Calculates the normalized Laplacian of a given adjacency matrix.

    Args:
        A: rank 2 array or sparse matrix
        symmetric: bool, a sign of whether to calculate symmetric normalization

    Returns:
        normalized laplacian
    """
    if sp.issparse(A):
        result_eye = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        result_eye = np.eye(A.shape[-1], dtype=A.dtype)

    normalized_adjacency_parameter = normalized_adjacency_matrix(
        A,
        symmetric=symmetric
    )

    return result_eye - normalized_adjacency_parameter
