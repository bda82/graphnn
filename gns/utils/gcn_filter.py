import copy
import warnings
from typing import Any

import numpy as np
from scipy import sparse as sp

from gns.utils.normalized_adjacency import normalized_adjacency


def gcn_filter(A, symmetric=True) -> Any:
    """
    Filters garf.

    Args:
        A: array or sparse matrix with rank 2 or 3
        symmetric: bool, a sign that the matrix needs to be normalized
    
    Returns:
        an array or a sparse matrix with rank 2 or 3, the same properties as the parameter A;
    """
    # Скопируем матрицу для исключения сайд-эффектв
    out = copy.deepcopy(A)

    if isinstance(A, list) or (isinstance(A, np.ndarray) and A.ndim == 3):
        for i in range(len(A)):
            out[i] = A[i]
            out[i][np.diag_indices_from(out[i])] += 1
            out[i] = normalized_adjacency(out[i], symmetric=symmetric)
    else:
        if hasattr(out, "tocsr"):
            out = out.tocsr()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[np.diag_indices_from(out)] += 1

        out = normalized_adjacency(out, symmetric=symmetric)

    if sp.issparse(out):
        out.sort_indices()

    return out
