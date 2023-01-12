import logging

from tensorflow.keras import backend as K  # noqa

from gns.config.settings import settings_fabric
from gns.layer.convolution import ConvolutionalGeneralLayer
from gns.utils.normalized_adjacency import normalized_adjacency
from gns.utils.modal_dot import modal_dot

settings = settings_fabric()

logger = logging.getLogger(__name__)


class GCSConvolutionalGeneralLayer(ConvolutionalGeneralLayer):
    """
    A special `GraphConv` layer with a trainable skip connection.

    Models:
        single
        disjoint
        mixed
        batch

    Input parameters:
        Node features of shape `([batch], n_nodes, n_node_features)`;
        Normalized adjacency matrix of shape `([batch], n_nodes, n_nodes)`
        (can be computed with `gns.utils.normalized_adjacency`)

    Output parameters:
        Node features with the same shape as the input, but with the last dimension changed to `channels`.
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
            `channels`: number of output channels;
            `activation`: activation function;
            `use_bias`: bool, add a bias vector to the output;
            `kernel_initializer`: initializer for the weights;
            `bias_initializer`: initializer for the bias vector;
            `kernel_regularizer`: regularization applied to the weights;
            `bias_regularizer`: regularization applied to the bias vector;
            `activity_regularizer`: regularization applied to the output;
            `kernel_constraint`: constraint applied to the weights;
            `bias_constraint`: constraint applied to the bias vector.
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

    def build(self, input_shape):
        """
        Build layer.

        Args:
            input_shape: input shape

        Returns:

        """
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        logger.info("Create the first kernel")

        self.kernel_1 = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name=settings.names.kernel_1,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        logger.info("Create the second kernel")

        self.kernel_2 = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name=settings.names.kernel_2,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # If we need a bias

        if self.use_bias:
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

        output = K.dot(x, self.kernel_1)
        output = modal_dot(a, output)
        skip = K.dot(x, self.kernel_2)
        output += skip

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a):
        return normalized_adjacency(a)


def gsn_convolutional_general_layer_fabric(
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
    return GCSConvolutionalGeneralLayer(
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
