import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from gns.dataset.dataset import Dataset
from gns.graph.graph import Graph
from gns.loaders.batch_loader import BatchLoader, batch_loader_fabric
from gns.loaders.single_loader import SingleLoader, single_loader_fabric
from gns.model.gcn import (
    GraphConvolutionalNetworkModel,
    graph_convolutional_network_model_fabric,
)

tf.keras.backend.set_floatx("float64")

batch_size = 16
n_nodes = 11
n_node_features = 7
n_edge_features = 3
n_labels = 32


def graph_fabric(n_nodes, n_features, n_edge_features=None, sparse=False):
    x = np.random.rand(n_nodes, n_features)
    a = np.random.randint(0, 2, (n_nodes, n_nodes)).astype("f4")
    e = (
        np.random.rand(np.count_nonzero(a), n_edge_features)
        if n_edge_features is not None
        else None
    )
    if sparse:
        a = sp.csr_matrix(a)
    return Graph(x=x, a=a, e=e)


class TDS(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        super().__init__()

    def read(self):
        return self.graphs


def test_graph_conv_model_single():
    dataset = TDS(
        [
            graph_fabric(
                n_nodes=n_nodes,
                n_features=n_node_features,
                n_edge_features=None,
                sparse=True,
            )
        ]
    )

    loader: SingleLoader = single_loader_fabric(dataset, epochs=1)

    inputs = list(loader)[0]

    model_instance: GraphConvolutionalNetworkModel = (
        graph_convolutional_network_model_fabric(**{"n_labels": n_labels})
    )

    output = model_instance(inputs)

    assert isinstance(output, tf.Tensor)


def test_graph_conv_model_batch():
    dataset = TDS(
        [
            graph_fabric(
                n_nodes=n_nodes,
                n_features=n_node_features,
                n_edge_features=None,
            )
            for _ in range(batch_size)
        ]
    )

    loader: BatchLoader = batch_loader_fabric(dataset, epochs=1, batch_size=batch_size)

    inputs = loader.__next__()

    model_instance: GraphConvolutionalNetworkModel = (
        graph_convolutional_network_model_fabric(**{"n_labels": n_labels})
    )

    output = model_instance(inputs)

    assert isinstance(output, tf.Tensor)
