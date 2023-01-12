from scipy import sparse as sp

from gns.utils.sp_matrix_to_sp_tensor import sp_matrix_to_sp_tensor


def sp_matrices_to_sp_tensors(inputs) -> tuple:
    """
    Converting Scipy sparse matrices to tensor.

    Args:
        inputs: inputs to convert

    Returns:
        converted data
    """
    inputs = list(inputs)

    for i in range(len(inputs)):
        if sp.issparse(inputs[i]):
            inputs[i] = sp_matrix_to_sp_tensor(inputs[i])
    
    return tuple(inputs)
