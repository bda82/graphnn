import copy
import warnings
from typing import Any

import numpy as np
from scipy import sparse as sp

from gns.utils.normalized_adjacency_matrix import normalized_adjacency_matrix
from gns.config.settings import settings_fabric

settings = settings_fabric()


def gcn_filter(A, symmetric=True) -> Any:
    """
    Filters garf.

    Args:
        A: array or sparse matrix with rank 2 or 3
        symmetric: bool, a sign that the matrix needs to be normalized
    
    Returns:
        an array or a sparse matrix with rank 2 or 3, the same properties as the parameter A;
    """
    # copy data to avoid some side effects

    result = copy.deepcopy(A)

    if_a_is_list_predicate = isinstance(A, list)
    if_a_is_numpy_ndarray_predicate = isinstance(A, np.ndarray)

    if if_a_is_list_predicate or (if_a_is_numpy_ndarray_predicate and A.ndim == 3):
        for i in range(len(A)):
            result[i] = A[i]
            result[i][np.diag_indices_from(result[i])] += 1
            result[i] = normalized_adjacency_matrix(result[i], symmetric=symmetric)
    else:
        if hasattr(result, settings.attribute_properties.tocsr):
            result = result.tocsr()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result[np.diag_indices_from(result)] += 1

        result = normalized_adjacency_matrix(result, symmetric=symmetric)

    issparse_predicate = sp.issparse(result)

    if issparse_predicate:
        result.sort_indices()

    return result
