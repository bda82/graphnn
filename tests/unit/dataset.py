import numpy as np
import pytest

from gns.dataset.dataset import Dataset
from gns.graph.graph import Graph, graph_fabric

n_node_features = 3
n_edge_features = 3
n_labels = 2
n_graphs = 10
n_graph_huge = 100


class TDS(Dataset):
    def read(self):
        return [
            graph_fabric(
                x=np.random.rand(n, n_node_features),
                a=np.random.randint(0, 2, (n, n)),
                e=np.random.rand(n, n, n_edge_features),
                y=np.array([0.0, 1.0]),
            )
            for n in np.random.randint(3, 8, n_graphs)
        ]


@pytest.fixture
def tds():
    return TDS()


@pytest.fixture
def graph():
    return graph_fabric(
        x=np.random.rand(n_graph_huge, n_node_features),
        a=np.random.randint(0, 2, (n_graph_huge, n_graph_huge)),
        e=np.random.rand(n_graph_huge, n_graph_huge, n_edge_features),
        y=np.array([0.0, 1.0]),
    )


def test_dataset_creation(tds):
    assert tds.n_node_features == n_node_features
    assert tds.n_edge_features == n_edge_features
    assert tds.n_labels == n_labels
    assert tds.__len__() == n_graphs


def test_dataset_attributes(tds):
    for k in ["x", "a", "e", "y"]:
        assert k in tds.signature

    assert isinstance(tds[0], Graph)
    assert isinstance(tds[:3], Dataset)
    assert isinstance(tds[[1, 3, 4]], Dataset)


def test_dataset_single_assignment(tds, graph):
    tds[0] = graph
    assert tds[0].n_nodes == n_graph_huge and all(
        [tds_.n_nodes != n_graph_huge for tds_ in tds[1:]]
    )


def test_dataset_slice_assignment(tds, graph):
    tds[1:3] = [graph, graph]
    assert (
        tds[1].n_nodes == n_graph_huge
        and tds[2].n_nodes == n_graph_huge
        and all([tds_.n_nodes != n_graph_huge for tds_ in tds[3:]])
    )


def test_dataset_sum(tds):
    tds2 = TDS()

    assert len(tds + tds2) == len(tds) + len(tds2)
    assert len(tds + tds2) == len(tds2 + tds)


def test_dataset_numpy_shuffle_dont_crashes(tds):
    try:
        np.random.shuffle(tds)
    except Exception as ex:
        assert False, f"np.random.shuffle(tds) raised an exception: {ex}"
