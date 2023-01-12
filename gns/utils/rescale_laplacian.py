import numpy as np
from scipy import linalg
from scipy import sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import ArpackNoConvergence


def rescale_laplacian(L, lmax=None) -> coo_matrix:
    """
    Scales the Laplace eigenvalues in `[-1,1]`,
    using lmax as the largest eigenvalue.
    
    Args:
        L: rank 2 array or sparse metric
        lmax: if None, then calculate the largest eigenvalue using `scipy.linalg.eigh`.
    
    If the native composition fails, the lmax is automatically set to 2.
    If it is a scalar, then we will use this value as the largest eigenvalue when scaling.
    
    Returns:
        laplacian
    """
    if lmax is None:
        try:
            if sp.issparse(L):
                lmax = sp.linalg.eigsh(L, 1, which="LM", return_eigenvectors=False)[0]
            else:
                n = L.shape[-1]
                lmax = linalg.eigh(L, eigvals_only=True, eigvals=[n - 2, n - 1])[-1]
        except ArpackNoConvergence:
            lmax = 2

    if sp.issparse(L):
        I = sp.eye(L.shape[-1], dtype=L.dtype)
    else:
        I = np.eye(L.shape[-1], dtype=L.dtype)

    L_scaled = (2.0 / lmax) * L - I

    return L_scaled
