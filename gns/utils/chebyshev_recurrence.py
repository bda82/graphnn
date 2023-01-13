import numpy as np

from scipy import sparse as sp


def chebyshev_recurrence(polynomial_minus_one, polynomial_minus_two, matrix) -> int | None:
    """
    Calculate chebyshev recurrence.
    """
    sparse_issparse_predicate = sp.issparse(matrix)

    x_matrix = list()

    if sparse_issparse_predicate:
        x_matrix = sp.csr_matrix(matrix, copy=True)
    else:
        x_matrix = np.copy(matrix)

    return 2 * x_matrix.dot(polynomial_minus_one) - polynomial_minus_two
