from tensorflow.keras.layers import Layer  # noqa

from gns.utils.check_dtypes_decorator import check_dtypes_decorator
from gns.utils.deserialize_kwarg import deserialize_kwarg
from gns.utils.is_keras_kwarg import is_keras_kwarg
from gns.utils.is_layer_kwarg import is_layer_kwarg
from gns.utils.serialize_kwarg import serialize_kwarg


class ConvolutionalGeneralLayer(Layer):
    """
    The main class for the convolutional layer of a Graph neural network.

    It is used as a base for expanding and creating your own
    classes - representations of GNS layers that implement standard matrix multiplications

    This is useful if you want to create layers that support dense input data,
    batch and mixed modes or other non-standard processing. The input data is not checked,
    which provides maximum flexibility.

    All class extensions must use the `call(self, inputs)` and `config(self)` methods.

    Notes:

    Input parameters are passed using arguments `**kwargs`.

    These can be additional arguments for Keras layers of the regulator type,
    initializers, constraints, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})

        self.supports_masking = True
        self.kwargs_keys = []

        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        self.call = check_dtypes_decorator(self.call)

    def build(self, input_shape) -> None:
        """Just set property now."""
        self.built = True

    def call(self, inputs):
        """
        Should be redefined.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """
        Combine configs.
        """
        base_config = super().get_config()

        keras_config = {}

        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))

        return {**base_config, **keras_config, **self.config}

    @property
    def config(self) -> dict:
        return {}

    @staticmethod
    def preprocess(a):
        return a


def convolutional_general_layer_fabric(**kwargs):
    return ConvolutionalGeneralLayer(**kwargs)
