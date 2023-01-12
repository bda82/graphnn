import tensorflow as tf
from tensorflow import SparseTensor
from tensorflow.keras import backend as K  # noqa


def reshape(a, shape=None, name: str | None = None) -> SparseTensor:
    """
    Changes the shape according to the shape, automatically coping with the rarefaction.

    Args:
        a: tensor or SparseTensor
        shape: new shape
        name: operation name
        
    Returns:
        tensor or SparseTensor.
    """
    if K.is_sparse(a):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(a, shape=shape, name=name)
