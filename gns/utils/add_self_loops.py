import tensorflow as tf
from tensorflow import SparseTensor


def add_self_loops(a, fill=1.0) -> SparseTensor:
    """
    Adds loops to the specified adjacency matrix.
    Loops are added only to those nodes that do not have them yet, they are added all according to
    the value of the fill parameter fill.

    Args:
        a: square SparseTensor.
        fill: the fill parameter for new loops, it will be converted to the `dtype` type for the `a` attribute of the graph.
    
    Returns:
        SparseTensor with the same shape as the input attribute `a`.
    """
    tensor_shape = tf.shape(a, out_type=a.indices.dtype)[0]

    mask_od = a.indices[:, 0] != a.indices[:, 1]
    mask_sl = ~mask_od
    mask_od.set_shape([None])  # noqa
    mask_sl.set_shape([None])  # noqa

    indices_od = a.indices[mask_od]
    indices_sl = a.indices[mask_sl]

    values_sl = tf.fill((tensor_shape,), tf.cast(fill, a.values.dtype))
    values_sl = tf.tensor_scatter_nd_update(
        values_sl, indices_sl[:, 0:1], a.values[mask_sl]
    )

    indices_sl = tf.range(tensor_shape, dtype=a.indices.dtype)[:, None]
    indices_sl = tf.repeat(indices_sl, 2, -1)
    indices = tf.concat((indices_od, indices_sl), 0)

    values_od = a.values[mask_od]
    values = tf.concat((values_od, values_sl), 0)

    out = tf.SparseTensor(indices, values, (tensor_shape, tensor_shape))

    sparse_tensor = tf.sparse.reorder(out)

    return sparse_tensor
