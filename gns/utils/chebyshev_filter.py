import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix

from gns.utils.chebyshev_polynomial import chebyshev_polynomial
from gns.utils.normalized_adjacency_matrix import normalized_adjacency_matrix
from gns.utils.rescale_laplacian import rescale_laplacian


def chebyshev_filter(A, k, symmetric=True) -> list[csr_matrix | int | None]:
    """
    Implementation of the Chebyshev filter for a given adjacency matrix.

    Args:
        A: rank 2 array or rank 2 array sparse matrix
        k: int, Chebyshev polynomial order
        symmetric: bool, whether to normalize the matrix
    
    Returns:
        a list of k +1 arrays or sparse matrices with one element in for each degree of the polynomial
    """
    normalized_adjacency_parameter = normalized_adjacency_matrix(A, symmetric)

    sparse_eye = None

    if sp.issparse(A):
        sparse_eye = sp.eye(A.shape[0], dtype=A.dtype)
    else:
        sparse_eye = np.eye(A.shape[0], dtype=A.dtype)

    # Calculate laplacian

    laplacian = sparse_eye - normalized_adjacency_parameter

    # Scale laplacian

    laplacian_scaled = rescale_laplacian(laplacian)

    # Calculate chebyshev polynomial approximation

    chebyshev_polynomial_parameter = chebyshev_polynomial(laplacian_scaled, k)

    # Sort indexes

    if sp.issparse(chebyshev_polynomial_parameter[0]):
        for i in range(len(chebyshev_polynomial_parameter)):
            chebyshev_polynomial_parameter[i].sort_indices()

    return chebyshev_polynomial_parameter
