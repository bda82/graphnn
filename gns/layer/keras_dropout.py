import tensorflow as tf


def keras_dropout_fabric(rate, noise_shape=None, seed=None, **kwargs):
    return tf.keras.layers.Dropout(
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
        **kwargs
    )
