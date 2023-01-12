import logging

import tensorflow as tf
from tensorflow import IndexedSlices, SparseTensor, Tensor

logger = logging.getLogger(__name__)


def check_dtypes(inputs) -> list[Tensor | IndexedSlices | SparseTensor | object]:
    """
    Checking the data set type.

    Args:
        inputs: dataset
    
    Returns:

    """
    for value in inputs:
        if not hasattr(value, "dtype"):
            return inputs

    if len(inputs) == 2:
        x, a = inputs
        e = None
    elif len(inputs) == 3:
        x, a, e = inputs
    else:
        return inputs

    valid_types = (
        tf.float16,
        tf.float32,
        tf.float64,
    )
    if a.dtype in (tf.int32, tf.int64) and x.dtype in valid_types:
        logger.warning(
            f"A sparse matrix of type (d type) {a.dtype} is incompatible with the dtype "
            f" type of node features {x.dtype} and will be automatically converted to type "
            f"{x.dtype}."
        )
        a = tf.cast(a, x.dtype)

    output = [_ for _ in [x, a, e] if _ is not None]
    return output
