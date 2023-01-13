from tensorflow.keras import backend as KerasBackend  # noqa
from tensorflow.keras.layers import Dense, Layer  # noqa

from gns.config.settings import settings_fabric

settings = settings_fabric()


class GlobalPoolLayer(Layer):
    """
    The base class of the layer for GlobalPool.

    Notes:
        Parameters of type pooling_op, batch_pooling_op must be redefined in
        specific implementations of the class before calling the `call()` method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True
        self.pooling_op = None
        self.batch_pooling_op = None
        self.data_mode = settings.models.disjoint

    def build(self, input_shape):
        """
        Build layer
        Args:
            input_shape: input shape

        Returns:

        """
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = settings.models.disjoint
        else:
            if len(input_shape) == 2:
                self.data_mode = settings.models.single
            else:
                self.data_mode = settings.models.batch

        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == settings.models.disjoint:
            X = inputs[0]
            I = inputs[1]
            if KerasBackend.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == settings.models.disjoint:
            return self.pooling_op(X, I)
        else:
            return self.batch_pooling_op(
                X,
                axis=-2,
                keepdims=(self.data_mode == settings.models.single)
            )

    def compute_output_shape(self, input_shape):
        """
        Calculate the output form.

        Args:
            input_shape: input form

        Returns:

        """
        if self.data_mode == settings.models.single:
            return (1,) + input_shape[-1:]
        elif self.data_mode == settings.models.batch:
            return input_shape[:-2] + input_shape[-1:]
        else:
            return input_shape[0]


def global_pool_layer_fabric(**kwargs):
    return GlobalPoolLayer(**kwargs)
