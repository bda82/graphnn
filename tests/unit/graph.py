import numpy as np
import pytest

from gns.graph.graph import graph_fabric

n_nodes = 5
n_node_features = 4
n_edge_features = 3
n_outs = 2


@pytest.fixture
def x():
    return np.ones((n_nodes, n_node_features))


@pytest.fixture
def a():
    return np.ones((n_nodes, n_nodes))


@pytest.fixture
def e():
    return np.ones((n_nodes, n_nodes, n_edge_features))


@pytest.fixture
def y():
    return np.ones((n_outs,))


@pytest.fixture
def graph(x, a, e, y):
    return graph_fabric(x=x, a=a, e=e, y=y)


def test_graph_creation_empty():
    try:
        graph_fabric()
    except Exception as ex:
        assert False, f"Empty Graph creation raises an error: {ex}"


def test_graph_creation_only_nodes(x):
    try:
        graph_fabric(x=x)
    except Exception as ex:
        assert False, f"Only node features Graph creation raises an error: {ex}"


def test_graph_creation_only_adjacency(a):
    try:
        graph_fabric(a=a)
    except Exception as ex:
        assert False, f"Only adjacency Graph creation raises an error: {ex}"


def test_graph_creation_all_parameters(x, a, e, y):
    try:
        graph_fabric(x=x, a=a, e=e, y=y)
    except Exception as ex:
        assert False, f"Full parameters Graph creation raises an error: {ex}"


def test_graph_creation_attributes(graph, a):
    assert graph.n_nodes == n_nodes
    assert graph.n_node_features == n_node_features
    assert graph.n_edge_features == n_edge_features
    assert graph.n_labels == n_outs
    assert graph.n_edges == np.count_nonzero(a)


def test_graph_get_numpy_parameters(graph):
    graph_numpy = graph.numpy()
    assert graph_numpy


def test_graph_gt_names_and_keys(graph, x, a, e, y):
    graph_gt = [x, a, e, y]
    graph_gt_names = ["x", "a", "e", "y"]

    graph_numpy = graph.numpy()

    for i in range(len(graph_gt)):
        assert np.all(graph_numpy[i] == graph_gt[i])

    for i in range(len(graph_gt)):
        assert np.all(graph.__getitem__(graph_gt_names[i]) == graph_gt[i])
