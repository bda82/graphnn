from typing import Type

import tensorflow as tf
from scipy import sparse as sp
from tensorflow import SparseTensorSpec, TensorSpec


def get_spec(x) -> Type[SparseTensorSpec | TensorSpec]:
    """
    Returns a specification (description or metadata) for a tensor of the tenzorflow type.Tensor.

    Args:
        x: object for explain

    Returns:
        explanation
    """
    if isinstance(x, tf.SparseTensor) or sp.issparse(x):
        return tf.SparseTensorSpec
    else:
        return tf.TensorSpec
