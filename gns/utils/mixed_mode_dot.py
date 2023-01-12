import tensorflow as tf
from tensorflow import IndexedSlices, SparseTensor, Tensor

from gns.utils.dot import dot
from gns.utils.reshape import reshape
from gns.utils.transpose import transpose


def mixed_mode_dot(a, b) -> object | Tensor | None | IndexedSlices | SparseTensor:
    """
    Calculates the equivalent of the tf.einsam function `('ij, bjk->bik', a, b)`, but
    works for both dense and sparse input data.

    Args:
        a: tensor or SparseTensor with rank 2.
        b: tensor or SparseTensor with rank 3.
        
    Returns:
        tensor or SparseTensor with rank 3.
    """
    a_shp = tf.shape(a)
    b_shp = tf.shape(b)

    b_t = transpose(b, (1, 2, 0))
    b_t = reshape(b_t, tf.stack((b_shp[1], -1)))
    output = dot(a, b_t)
    output = reshape(output, tf.stack((a_shp[0], b_shp[2], -1)))
    output = transpose(output, (2, 0, 1))

    return output
