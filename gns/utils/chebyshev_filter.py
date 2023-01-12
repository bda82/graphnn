import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix

from gns.utils.chebyshev_polynomial import chebyshev_polynomial
from gns.utils.normalized_adjacency import normalized_adjacency
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
    normalized_adj = normalized_adjacency(A, symmetric)
    if sp.issparse(A):
        I = sp.eye(A.shape[0], dtype=A.dtype)
    else:
        I = np.eye(A.shape[0], dtype=A.dtype)

    # Вычисление лапласиана

    L = I - normalized_adj

    # Масштабирование лапласиана
    L_scaled = rescale_laplacian(L)

    # Вычисление аппроксимации полинома Чебышева

    T_k = chebyshev_polynomial(L_scaled, k)

    # Сортировка индексов

    if sp.issparse(T_k[0]):
        for i in range(len(T_k)):
            T_k[i].sort_indices()

    return T_k
