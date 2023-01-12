import tensorflow as tf
from tensorflow.keras import backend as KerasBackend  # noqa


def transpose(a, perm=None, name: str | None = None):
    """
    Transposes parameter a, automatically coping with the rarefaction with the help of overloaded
    functions of the Flow of Ten worlds.
    
    Args:
        a: tensor or Sparse sensor with rank k.
        perm: permutation indices of size k.
        name: operation name.
    
    Returns:
        Tensor or sparse tensor with rank k.
    """
    if KerasBackend.is_sparse(a):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        # Default value if the form is set to empty
        perm = (1, 0)
        
    return transpose_op(a, perm=perm, name=name)
