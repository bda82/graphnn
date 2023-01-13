import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix

from gns.utils.chebyshev_recurrence import chebyshev_recurrence


def chebyshev_polynomial(X, k) -> list[csr_matrix | int | None]:
    """
    Computes Chebyshev polynomials from X up to the order of k.

    Args:
        X: rank 2 array or sparse matrix
        k: the order to which the polynomials are calculated
    
    Returns:
        a list of k +1 arrays or sparse matrices with one element for each degree of the polynomial.
    """
    polynomial = list()

    sparse_eye = sp.eye(X.shape[0], dtype=X.dtype)
    numpy_eye = np.eye(X.shape[0], dtype=X.dtype)

    sparse_issparse_predicate = sp.issparse(X)

    if sparse_issparse_predicate:
        polynomial.append(sparse_eye.tocsr())
    else:
        polynomial.append(numpy_eye)

    polynomial.append(X)

    for _ in range(2, k + 1):
        polynomial.append(
            chebyshev_recurrence(
                polynomial[-1],
                polynomial[-2],
                X
            )
        )

    return polynomial
