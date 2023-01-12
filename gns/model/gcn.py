import tensorflow as tf

from gns.config.settings import settings_fabric
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer

settings = settings_fabric()


class GraphConvolutionalNetworkModel(tf.keras.Model):
    """
    The main model of a convolutional neural network, complementing the Tensotflow/Keras model.

    Operating modes:
        single
        disjoint
        mixed
        batch

    Input parameters:
        features of shape nodes `([batch], n_nodes, n_node_features)`
        weighted adjacency matrix of shape `([batch], n_nodes, n_nodes)`

    Output parameters:
        Softmax forecasts with the form `([batch], n_nodes, n_labels)`
    """

    def __init__(
        self,
        n_labels,
        channels=16,
        activation=settings.activations.relu,
        output_activation=settings.activations.softmax,
        use_bias=False,
        dropout_rate=0.5,
        l2_reg=2.5e-4,
        **kwargs,
    ):
        """
        Args:
            n_labels: number of channels to output
            channels: a set of channels for the first GCNConvolutionalGeneralLayer layer
            activation: activation function for the first GCNConvolutionalGeneralLayer layer
            output_activation: activation function for the second GCNConvolutionalGeneralLayer layer
            use_bias: indicates the use of the bias vector for training for both GCNConvolutionalGeneralLayer layers
            dropout_rate: the rate used in dropout layers (Dropout)
            l2_reg: regularization strength l2
            **kwargs: additional attributes passed to the base model constructor (`tf.keras.Model.__init__`).
        """
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        reg = tf.keras.regularizers.l2(l2_reg)

        self._d0 = tf.keras.layers.Dropout(dropout_rate)

        self._gcn0 = GCNConvolutionalGeneralLayer(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )

        self._d1 = tf.keras.layers.Dropout(dropout_rate)

        self._gcn1 = GCNConvolutionalGeneralLayer(
            n_labels, activation=output_activation, use_bias=use_bias
        )

    def get_config(self) -> dict:
        return dict(
            n_labels=self.n_labels,
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )

    def call(self, inputs) -> GCNConvolutionalGeneralLayer:
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader

        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])


def graph_convolutional_network_model_fabric(
    n_labels,
    channels=16,
    activation=settings.activations.relu,
    output_activation=settings.activations.softmax,
    use_bias=False,
    dropout_rate=0.5,
    l2_reg=2.5e-4,
    **kwargs,
):
    return GraphConvolutionalNetworkModel(
        n_labels,
        channels,
        activation,
        output_activation,
        use_bias,
        dropout_rate,
        l2_reg,
        **kwargs,
    )
