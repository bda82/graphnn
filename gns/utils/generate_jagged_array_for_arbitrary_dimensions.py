import numpy as np


def generate_jagged_array_for_arbitrary_dimensions(x, target_shape):
    """
    Given a jagged array of arbitrary dimensions, zero-pads all elements in the
    array to match the provided `target_shape`.

    Args:
        x: a list or np.array of dtype object, containing np.arrays with variable dimensions;
        target_shape: a tuple or list s.t. `target_shape[i] >= x.shape[i]` for each x in `X`. 
                      If `target_shape[i] = -1`, it will be automatically converted to `X.shape[i]`, 
                      so that passing a target shape of e.g. `(-1, n, m)`
                      will leave the first  dimension of each element untouched.

    Returns:
        a `np.array` of shape `(len(x), ) + target_shape`.
    """
    if len(x) < 1:
        raise ValueError(
            f"list `x` should be empty: {x}."
        )

    target_shape = tuple(
        shp if shp != -1 else x[0].shape[j]
        for j, shp in enumerate(target_shape)
    )

    output = np.zeros(
        (len(x),) + target_shape,
        dtype=x[0].dtype
    )

    for i in range(len(x)):
        slc = (i,) + tuple(slice(shp) for shp in x[i].shape)
        output[slc] = x[i]

    return output
