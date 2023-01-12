import tensorflow as tf
from tensorflow import SparseTensor
from tensorflow.keras import backend as K  # noqa

from gns.utils.dot import dot
from gns.utils.mixed_mode_dot import mixed_mode_dot
from gns.utils.reshape import reshape
from gns.utils.transpose import transpose


def modal_dot(a, b, transpose_a: bool = False, transpose_b: bool = False):
    """
    Calculates matrix multiplication for a and b.
    Automatically processes models.

    Note:
    This is a wrapper for standard matrix multiplication operations for dimensions 2 and 3:
    
    Supports automatic broadcast of the "packet" dimension if two input parameters
    they have different ranks.
    
    Supports any combination of dense and sparse input data.

    We can use this operation to multiply matrices representing packets of graphs
    in various modes for which adjacency matrices may or may not be
    sparse and have ranks other than node attributes.

    In addition, the function also supports the case when we have many adjacency matrices
    and only one count.

    For example:
        - `a` of rank 2, `b` of rank 2 -> calculate `a @ b`
        - `a` of rank 3, `b` of rank 3 -> calculate `[a[i] @ b[i] for i in range(len(a))]`
        - `a` of rank 2, `b` of rank 3 -> calculate `[a @ b[i] for i in range(len(b))]`
        - `a` of rank 3, `b` rank 2 -> calculate `[a[i] @ b for i in range(len(a))]`

    Args:
        a: tensor or SparseTensor with rank 2 or 3;
        b: tensor or SparseTensor with rank 2 or 3;
        transpose_a: indicates whether the internal dimension 2 of parameter a needs to be transposed;
        transpose_b: indicates the need to transpose the internal dimension 2 of parameter b;

    Returns: 
        tensor or SparseTensor with `rank = max(rank(a), rank(b))`.
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)

    if a_ndim not in (2, 3):
        raise ValueError(f"The parameter `a` with rank 2 or 3 is expected, but the dimension {a_ndim} is obtained")

    if b_ndim not in (2, 3):
        raise ValueError(f"The parameter `b` with rank 2 or 3 is expected, but the dimension {b_ndim} is obtained")

    # If you need transposition by dimensions

    if transpose_a:
        perm = None if a_ndim == 2 else (0, 2, 1)
        a = transpose(a, perm)

    if transpose_b:
        perm = None if b_ndim == 2 else (0, 2, 1)
        b = transpose(b, perm)

    if a_ndim == b_ndim:
        # Case of ...ij,...jk->...ik
        return dot(a, b)
    elif a_ndim == 2:
        # Case of ij,bjk->bik
        return mixed_mode_dot(a, b)
    else:
        # Here will be a_ndim == 3
        # Case bij,jk->bik
        if not K.is_sparse(a) and not K.is_sparse(b):
            # No need to change the shape - let's go back to the standard dense matrix multiplication
            return tf.matmul(a, b)

        # If any of the input data is sparse, we use the `dot(a,b)`` function
        a_shape = tf.shape(a)
        b_shape = tf.shape(b)
        a_flat = reshape(a, (-1, a_shape[2]))
        output = dot(a_flat, b)

        return reshape(output, (-1, a_shape[1], b_shape[1]))
