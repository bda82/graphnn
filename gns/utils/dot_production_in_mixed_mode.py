import tensorflow as tf
from tensorflow import IndexedSlices, SparseTensor, Tensor

from gns.utils.dot_production import dot_production
from gns.utils.reshape import reshape
from gns.utils.transpose import transpose


def dot_production_in_mixed_mode(a, b) -> object | Tensor | None | IndexedSlices | SparseTensor:
    """
    Calculates the equivalent of the `tf.einsum` (Tensor contraction over specified indices and outer product)
    function `('ij, bjk->bik', a, b)` (works for both dense and sparse input data).

    Args:
        a: tensor or SparseTensor with rank 2.
        b: tensor or SparseTensor with rank 3.
        
    Returns:
        tensor or SparseTensor with rank 3.
    """
    shape_a = tf.shape(a)
    shape_b = tf.shape(b)

    transposed_b = transpose(b, (1, 2, 0))
    transposed_b = reshape(transposed_b, tf.stack((shape_b[1], -1)))

    result = dot_production(a, transposed_b)

    result = reshape(result, tf.stack((shape_a[0], shape_b[2], -1)))
    result = transpose(result, (2, 0, 1))

    return result
