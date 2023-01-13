from scipy import sparse as sp

from gns.utils.sparse_matrix_to_sparse_tensor import sparse_matrix_to_sparse_tensor


def sparse_matrices_to_sparse_tensors(inputs) -> tuple:
    """
    Converting Scipy sparse matrices to tensor.

    Args:
        inputs: inputs to convert

    Returns:
        converted data
    """
    inputs = list(inputs)

    for i in range(len(inputs)):
        issparse_predicate = sp.issparse(inputs[i])
        if issparse_predicate:
            inputs[i] = sparse_matrix_to_sparse_tensor(inputs[i])
    
    return tuple(inputs)
