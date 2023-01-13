from typing import Any
import logging

from tensorflow.keras import backend as KerasBackend  # noqa

from gns.config.settings import settings_fabric
from gns.layer.convolution import ConvolutionalGeneralLayer
from gns.utils.dot_production_modal import dot_production_modal
from gns.utils.normalized_laplacian import normalized_laplacian
from gns.utils.rescale_laplacian import rescale_laplacian

settings = settings_fabric()

logger = logging.getLogger(__name__)


class ChebyshevConvolutionalLayer(ConvolutionalGeneralLayer):
    """
    Chebyshev convolutional layer for a graph neural network.

    Operating modes: `single`, `disjoint`, `mixed`, `batch`.

    At the input of the layer are submitted:
        features of nodes of the form `([batch], n_nodes, n_node_features)`
        list of Chebyshev polynomials (KerasBackend) of the form
            `[([batch], n_nodes, n_nodes), ..., ([batch], n_nodes, n_nodes)]`
            can be calculated using the function `gns.utils.chebyshev_filter`.

    At the output of the layer:
        features nodes with the same shape as at the input, but with the last dimension changed to `channels`.

    """

    def __init__(
        self,
        channels,
        K=1,
        activation=None,
        use_bias=True,
        kernel_initializer=settings.initializers.glorot_uniform,
        bias_initializer=settings.initializers.zeros,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        """
        Args:
            channels: number of output channels
            K: KerasBackend: Chebyshev polynomial order
            activation: activation function
            use_bias: bool, add bias vector to output or not
            kernel_initializer: initializer for weights
            bias_initializer: initializer for bios vector
            kernel_regularizer: regularization applied to weights
            bias_regularizer: regularization applied to the bias vector
            activity_regularizer: regularization applied to weights outputs
            kernel_constraint: constraint applied to weights;
            bias_constraint: constraint applied to the bias vector
            **kwargs: additional parameters
        """
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.K = K

    def build(self, input_shape) -> None:
        if len(input_shape) < 2:
            raise ValueError(
                f"Wrong input form {input_shape}."
            )

        input_dim = input_shape[0][-1]

        logger.info("Add a kernel.")

        self.kernel = self.add_weight(
            shape=(self.K, input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # If bias is available

        if self.use_bias:
            logger.info("Build bias.")
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name=settings.names.bias,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.built = True

    def call(self, inputs, mask=None) -> Any:
        """
        Call calculations.

        Args:
            inputs: inputs
            mask: mask

        Returns:

        """
        x, a = inputs

        producted_0 = x

        output = KerasBackend.dot(producted_0, self.kernel[0])

        if self.K > 1:
            producted_1 = dot_production_modal(a, x)
            output += KerasBackend.dot(producted_1, self.kernel[1])

        for k in range(2, self.K):
            producted_2 = 2 * dot_production_modal(a, producted_1) - producted_0
            output += KerasBackend.dot(producted_2, self.kernel[k])
            producted_0, producted_1 = producted_1, producted_2

        # If bias enabled

        if self.use_bias:
            output = KerasBackend.bias_add(output, self.bias)

        if mask is not None:
            output *= mask[0]

        return self.activation(output)

    @property
    def config(self) -> dict:
        return {
            "channels": self.channels,
            "KerasBackend": self.K
        }

    @staticmethod
    def preprocess(a):
        """
        Preprocessing.

        Args:
            a: connectivity matrix

        Returns:

        """
        a = normalized_laplacian(a)

        # Scaling the Laplace eigenvalue

        a = rescale_laplacian(a)

        return a


def chebyshev_convolutional_layer_fabric(
    channels,
    K=1,
    activation=None,
    use_bias=True,
    kernel_initializer=settings.initializers.glorot_uniform,
    bias_initializer=settings.initializers.zeros,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
):
    return ChebyshevConvolutionalLayer(
        channels,
        K,
        activation,
        use_bias,
        kernel_initializer,
        bias_initializer,
        kernel_regularizer,
        bias_regularizer,
        activity_regularizer,
        kernel_constraint,
        bias_constraint,
        **kwargs
    )
