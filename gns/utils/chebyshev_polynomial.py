import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix


def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X) -> int | None:
    if sp.issparse(X):
        X_ = sp.csr_matrix(X, copy=True)
    else:
        X_ = np.copy(X)

    return 2 * X_.dot(T_k_minus_one) - T_k_minus_two


def chebyshev_polynomial(X, k) -> list[csr_matrix | int | None]:
    """
    Computes Chebyshev polynomials from X up to the order of k.

    Args:
        X: rank 2 array or sparse matrix
        k: the order to which the polynomials are calculated
    
    Returns:
        a list of k +1 arrays or sparse matrices with one element for each degree of the polynomial.
    """
    T_k = list()

    if sp.issparse(X):
        T_k.append(sp.eye(X.shape[0], dtype=X.dtype).tocsr())
    else:
        T_k.append(np.eye(X.shape[0], dtype=X.dtype))

    T_k.append(X)

    for _ in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k
