import tensorflow as tf
import logging

from gns.config.settings import settings_fabric
from gns.message.generic_message_passing import GenericMessagePassing
from gns.utils.add_self_loops import add_self_loops

settings = settings_fabric()

logger = logging.getLogger(__name__)


class GraphSageConvolutionalLayer(GenericMessagePassing):
    """
    The main layer with the GraphSAGE algorithm.

    Operating modes:
        single
        disjoint
        mixed.

    Notes:
        The adjacency matrix must be sparse.

    Notes:
    The following aggregation methods are supported:
        sum
        mean
        max
        min
        product.

    Input parameters:
        Features of nodes in the form `(n_nodes, n_node_features)`
        Binary adjacency matrix of the form `(n_nodes, n_nodes)`

    Output parameters:
        Features of nodes of the same shape as the input parameter, but with the last
        measurement changed by the `channels` parameter.
    """

    def __init__(
        self,
        channels,
        aggregate="mean",
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
        `channels`: number of output channels
        `aggregate_op`: str, aggregation method ('sum', 'mean', 'max', 'min', 'prod');
        `activation': activation function
        `use_bias`: bool, a sign of adding bias vectors to outputs
        `kernel_initializer': initializer of weights
        `bias_initializer`: initializer of the bias vector
        `kernel_regularizer': a regularizer applied to weights
        `bias_regularizer': a regularizer applied to the bias vector
        `activity_regularizer': a regularizer applied to outputs
        `kernel_constraint': restrictions applied to weights
        `bias_constraint': constraints applied to the bias vector
        `**kwargs`: additional class attributes
        """
        super().__init__(
            aggregate=aggregate,
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

        self.built = False

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError(
                f"Invalid shape {input_shape}."
            )

        input_dim = input_shape[0][-1]

        logger.info("Build kernel.")

        self.kernel = self.add_weight(
            shape=(2 * input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # If bias enabled

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

    def call(self, inputs):  # noqa
        """
        Call layer.

        Args:
            inputs: inputs

        Returns:

        """
        x, a, _ = self.get_inputs(inputs)

        a = add_self_loops(a)

        aggregated = self.propagate(x, a)

        output = tf.keras.backend.concatenate([x, aggregated])

        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        output = tf.keras.backend.l2_normalize(output, axis=-1)

        if self.activation is not None:
            output = self.activation(output)

        return output

    @property
    def config(self) -> dict:
        return {"channels": self.channels}


def graph_sage_convolutional_layer_fabric(
    channels,
    aggregate=settings.aggregation_methods.mean,
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
    return GraphSageConvolutionalLayer(
        channels,
        aggregate,
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
