import os
import tensorflow as tf

from gns.config.settings import settings_fabric
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer, gcn_convolutional_general_layer_fabric
from gns.layer.keras_dropout import keras_dropout_fabric
from gns.model.model_folder import MODEL_FOLDER

settings = settings_fabric()


class GraphConvolutionalNetworkModel(tf.keras.Model):
    """
    The main model of a convolutional neural network, complementing the Tensorflow/Keras model.

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

        regularizer = tf.keras.regularizers.l2(l2_reg)

        self.dropout_layer_0: tf.keras.layers.Dropout = keras_dropout_fabric(dropout_rate)
        self.dropout_layer_1: tf.keras.layers.Dropout = keras_dropout_fabric(dropout_rate)

        self.convolutional_layer_0: GCNConvolutionalGeneralLayer = gcn_convolutional_general_layer_fabric(
            channels,
            activation=activation,
            kernel_regularizer=regularizer,
            use_bias=use_bias
        )
        self.convolutional_layer_1 = gcn_convolutional_general_layer_fabric(
            n_labels,
            activation=output_activation,
            use_bias=use_bias
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

    def call(self, inputs) -> GCNConvolutionalGeneralLayer:  # noqa
        if len(inputs) == 2:
            x, a = inputs
        else:
            # Model can be used with DisjointLoader
            x, a, _ = inputs

        x = self.dropout_layer_0(x)
        x = self.convolutional_layer_0([x, a])  # noqa
        x = self.dropout_layer_1(x)
        return self.convolutional_layer_1([x, a])  # noqa

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

def path():
    return os.path.join(MODEL_FOLDER, 'GraphConvolutionalNetworkModel')
