import logging

import tensorflow as tf
from tensorflow import IndexedSlices, SparseTensor, Tensor

from gns.config.settings import settings_fabric

logger = logging.getLogger(__name__)

settings = settings_fabric()


def check_dtypes(inputs) -> list[Tensor | IndexedSlices | SparseTensor | object]:
    """
    Checking the data set type.

    Args:
        inputs: dataset
    
    Returns:

    """
    for value in inputs:
        if not hasattr(value, settings.attribute_properties.dtype):
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
            f"Sparse matrix of type (dtype) {a.dtype}"
            f" incompatible with the type {settings.attribute_properties.dtype}"
            f"and node features {x.dtype} - will be converted (type cast) to"
            f" {x.dtype}."
        )

        a = tf.cast(a, x.dtype)

    return [_ for _ in [x, a, e] if _ is not None]
