import logging

from tensorflow.keras import backend as KeasBackend  # noqa

from gns.config.settings import settings_fabric
from gns.layer.convolution import ConvolutionalGeneralLayer
from gns.utils.gcn_filter import gcn_filter
from gns.utils.modal_dot import modal_dot

settings = settings_fabric()

logger = logging.getLogger(__name__)


class GCNConvolutionalGeneralLayer(ConvolutionalGeneralLayer):
    """
    Convolutional layer of a graph neural network.

    Models:
        single
        disjoint
        mixed
        batch

    Input parameters:
        Features of shape nodes `([batch], n_nodes, n_node_features)`
        Modified shape Laplacian `([batch], n_nodes, n_nodes)`, which can be calculated
            using the function `gnu.utils.gcn_filter`.

    Output parameters:
        Features of nodes with the same shape as at the input, but with loss of measurement,
        according to the channel parameters.
    """

    def __init__(
        self,
        channels,
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
            activation: activation function
            use_bias: bool, a sign of adding the bias vector to the outputs
            kernel_initializer: initializer of weights
            bias_initializer: initializer of the bias vector
            kernel_regularizer: a regularizer applicable to weights
            bias_regularizer: a regularizer applicable to the bias vector
            activity_regularizer: a regularizer applicable to outputs
            kernel_constraint: restrictions applicable to weights
            bias_constraint: constraints applicable to the bias vector
            kwargs: additional class arguments
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
        self.built = False
        self.channels = channels

    def build(self, input_shape) -> None:
        """
        Build layer.

        Args:
            input_shape: input shape

        Returns:

        """
        if len(input_shape) < 2:
            raise ValueError("Wrong input shape")

        input_dim = input_shape[0][-1]

        logger.info("Create kernel")

        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name=settings.names.kernel,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # If we need a bias

        if self.use_bias:
            logger.info("Build bias")
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name=settings.names.bias,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.built = True

    def call(self, inputs, mask=None):
        """
        Call layer.

        Args:
            inputs: inputs
            mask: mask

        Returns:

        """
        x, a = inputs

        output = KeasBackend.dot(x, self.kernel)
        output = modal_dot(a, output)

        if self.use_bias:
            output = KeasBackend.bias_add(output, self.bias)
        if mask is not None:
            output *= mask[0]

        output = self.activation(output)

        return output

    @property
    def config(self) -> dict:
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)


def gcn_convolutional_general_layer_fabric(
    channels,
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
    return GCNConvolutionalGeneralLayer(
        channels,
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
