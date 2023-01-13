import tensorflow as tf
from tensorflow import SparseTensor
from tensorflow.keras import backend as KerasBackend  # noqa


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
    if KerasBackend.is_sparse(a):
        reshape_result = tf.sparse.reshape
    else:
        reshape_result = tf.reshape

    return reshape_result(a, shape=shape, name=name)
