import numpy as np
import pytest
import scipy.sparse as sp

from gns.dataset.dataset import Dataset
from gns.graph.graph import graph_fabric
from gns.loaders.single_loader import SingleLoader, single_loader_fabric
from gns.loaders.batch_loader import BatchLoader, batch_loader_fabric

n_epochs = 1
n_node_features = 3
n_edge_features = 3
batch_size = 6
n_graphs = 10
N = 10
graphs_in_batch = n_graphs % batch_size


class TDS(Dataset):
    def read(self):
        return [
            graph_fabric(
                x=np.random.rand(N, n_node_features),
                a=sp.csr_matrix(np.random.randint(0, 2, (N, N))),
                e=np.random.rand(N, N, n_edge_features),
                y=np.array(N * [[0.0, 1.0]]),
            )
        ]


class TDSB(Dataset):
    ns = np.random.randint(3, 8, n_graphs)
    def read(self):
        return [
            graph_fabric(
                x=np.random.rand(n, n_node_features),
                a=sp.csr_matrix(np.random.randint(0, 2, (n, n))),
                e=np.random.rand(n, n, n_edge_features),
                y=np.array([0.0, 1.0]),
            )
            for n in self.ns
        ]


@pytest.fixture
def tds():
    return TDS()


@pytest.fixture
def tdsb():
    return TDSB()


def test_single(tds):
    n = tds.n_nodes

    loader: SingleLoader = single_loader_fabric(tds, sample_weights=np.ones(n), epochs=n_epochs)

    batches = list(loader)

    (x, a, e), y, sw = batches[0]

    signature = loader.tf_signature()

    assert (
        len(batches) == 1
        and x.shape == (n, n_node_features)
        and a.shape == (n, n)
        and len(e.shape) == 2
        and e.shape[-1] == n_edge_features
        and e.shape[0] == a.values.shape[0]
        and y.shape == (n, 2)
        and loader.steps_per_epoch == 1
        and len(signature[0]) == 3
    )


def test_batch(tdsb):
    loader: BatchLoader = batch_loader_fabric(tdsb, batch_size=batch_size, epochs=1, shuffle=False)
    batches = list(loader)

    (x, a, e), y = batches[-1]
    
    n = max(tdsb.ns[-graphs_in_batch:])

    signature = loader.tf_signature()

    assert (
        x.shape == (graphs_in_batch, n, n_node_features)
        and a.shape == (graphs_in_batch, n, n)
        and e.shape == (graphs_in_batch, n, n, n_edge_features)
        and y.shape == (graphs_in_batch, 2)
        and loader.steps_per_epoch == np.ceil(len(tdsb) / batch_size)
        and len(signature[0]) == 3
    )
