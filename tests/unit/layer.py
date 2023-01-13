import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import Input, Model  # noqa

from gns.config.settings import settings_fabric
from gns.layer.cheb import ChebyshevConvolutionalLayer
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer
from gns.layer.graphsage import GraphSageConvolutionalLayer
from gns.utils.sparse_matrix_to_sparse_tensor import sparse_matrix_to_sparse_tensor

settings = settings_fabric()

tf.keras.backend.set_floatx("float64")

MODES = {
    "SINGLE": 0,
    "BATCH": 1,
    "MIXED": 2,
}

N = 11
n_channels = 8
batch_size = 32
k_backend = 3
n_node_features = 7
n_edge_features = 3


A = np.ones((N, N))
X = np.random.normal(size=(N, n_node_features))
E = np.random.normal(size=(N, N, n_edge_features))


def test_graphsage_single():
    inputs = [Input(shape=(n_node_features,)), Input(shape=(None,), sparse=True)]

    input_data = [X, sparse_matrix_to_sparse_tensor(A)]

    layer_instance = GraphSageConvolutionalLayer(
        **{"channels": n_channels, "activation": settings.activations.relu}
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (N, n_channels)


def test_graphsage_mixed():
    inputs = [Input(shape=(N, n_node_features)), Input(shape=(N,), sparse=True)]

    input_data = [np.stack([X] * batch_size), sparse_matrix_to_sparse_tensor(A)]

    layer_instance = GraphSageConvolutionalLayer(
        **{"channels": n_channels, "activation": settings.activations.relu}
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, n_channels)


def test_conv_single():
    inputs = [Input(shape=(n_node_features,)), Input(shape=(None,), sparse=True)]

    input_data = [X, sparse_matrix_to_sparse_tensor(A)]

    layer_instance = GCNConvolutionalGeneralLayer(
        **{"channels": n_channels, "activation": settings.activations.relu}
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (N, n_channels)


def test_conv_mixed():
    X_batch = np.stack([X] * batch_size)

    inputs = [Input(shape=(N, n_node_features)), Input(shape=(N,), sparse=True)]

    input_data = [X_batch, sparse_matrix_to_sparse_tensor(A)]

    layer_instance = GCNConvolutionalGeneralLayer(
        **{"channels": n_channels, "activation": settings.activations.relu}
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, n_channels)


def test_chebishev_conv_single():
    inputs = [Input(shape=(n_node_features,)), Input(shape=(None,), sparse=True)]

    input_data = [X, sparse_matrix_to_sparse_tensor(A)]

    layer_instance = ChebyshevConvolutionalLayer(
        **{
            "KerasBackend": k_backend,
            "channels": n_channels,
            "activation": settings.activations.relu,
        }
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (N, n_channels)


def test_chebishev_conv_batch():
    inputs = [Input(shape=(N, n_node_features)), Input(shape=(N, N))]
    input_data = [np.stack([X] * batch_size), np.stack([A] * batch_size)]

    layer_instance = ChebyshevConvolutionalLayer(
        **{
            "KerasBackend": k_backend,
            "channels": n_channels,
            "activation": settings.activations.relu,
        }
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, n_channels)


def test_chebishev_conv_mixed():
    inputs = [Input(shape=(N, n_node_features)), Input(shape=(N,), sparse=True)]

    input_data = [np.stack([X] * batch_size), sparse_matrix_to_sparse_tensor(A)]

    layer_instance = ChebyshevConvolutionalLayer(
        **{
            "KerasBackend": 3,
            "channels": n_channels,
            "activation": settings.activations.relu,
        }
    )
    output = layer_instance(inputs)
    model = Model(inputs, output)

    output = model(input_data)

    assert output.shape == (batch_size, N, n_channels)
