import tensorflow as tf
from tensorflow.keras import backend as KerasBackend  # noqa

from gns.utils.dot_production import dot_production
from gns.utils.dot_production_in_mixed_mode import dot_production_in_mixed_mode
from gns.utils.reshape import reshape
from gns.utils.transpose import transpose


def dot_production_modal(a, b, transpose_a: bool = False, transpose_b: bool = False):
    """
    Calculates matrix multiplication for a and b.

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
    dimension_a = KerasBackend.ndim(a)
    dimension_b = KerasBackend.ndim(b)

    if dimension_a not in (2, 3):
        raise ValueError(f"The parameter `a` with rank 2 or 3 is expected, but the dimension {dimension_a} is obtained")

    if dimension_b not in (2, 3):
        raise ValueError(f"The parameter `b` with rank 2 or 3 is expected, but the dimension {dimension_b} is obtained")

    # If we need to transpose `a`

    if transpose_a:
        perm = None if dimension_a == 2 else (0, 2, 1)
        a = transpose(a, perm)

    # If we need to transpose `b`

    if transpose_b:
        perm = None if dimension_b == 2 else (0, 2, 1)
        b = transpose(b, perm)

    # Process dimensions

    if dimension_a == dimension_b:
        # Dimension ...ij,...jk->...ik
        return dot_production(a, b)
    elif dimension_a == 2:
        # Dimension ij,bjk->bik
        return dot_production_in_mixed_mode(a, b)
    else:
        # If dimension_a == 3 - dimension be bij,jk->bik
        if not KerasBackend.is_sparse(a) and not KerasBackend.is_sparse(b):
            return tf.matmul(a, b)

        shape_a = tf.shape(a)
        shape_b = tf.shape(b)

        flat_a = reshape(a, (-1, shape_a[2]))

        result_production = dot_production(flat_a, b)
        result_production = reshape(result_production, (-1, shape_a[1], shape_b[1]))

        return result_production
