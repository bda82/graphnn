import numpy as np
import tensorflow as tf
from scipy import sparse as sp
from tensorflow import SparseTensor


def sp_matrix_to_sp_tensor(x) -> SparseTensor:
    """
    Converts a sparse Scipy matrix into a sparse tensor.
    
    The output data indexes are reordered in the canonical row order, and
    duplicate entries are summed together (which is the default behavior of Scipy).

    Args:
        x: Scipy sparse matrix.
    
    Returns: 
        SparseTensor.
    """
    if len(x.shape) != 2:
        raise ValueError("x must have rank 2")

    row, col, values = sp.find(x)

    out = tf.SparseTensor(
        indices=np.array([row, col]).T, values=values, dense_shape=x.shape
    )
    
    return tf.sparse.reorder(out)
