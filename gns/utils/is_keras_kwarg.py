KERAS_KWARGS = {
    "trainable",
    "name",
    "dtype",
    "dynamic",
    "input_dim",
    "input_shape",
    "batch_input_shape",
    "batch_size",
    "weights",
    "activity_regularizer",
    "autocast",
    "implementation",
}


def is_keras_kwarg(key):
    return key in KERAS_KWARGS
