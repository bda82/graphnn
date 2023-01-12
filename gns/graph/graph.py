import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

import logging

logger = logging.getLogger(__name__)


class Graph:
    """
    The main container for Graph representation.
    The data associated with this class should be distributed according to the following attributes:
        `x`: for node features
        `a`: for adjacency matrix
        `e`: for graph edge attributes
        `y`: for nodes or graph labels

    All of these parameters are set to None by default, unless otherwise defined
    in the constructor of the class when creating an instance.

    If you need to read all non-empty attributes at a time (not None), then you can use
    using the numpy() method, which will return all the data in the form of a tuple in the order `x`, `a`, `e`, `y`.

    The graph object also has the following attributes, which are calculated automatically based on the data:
        `n_nodes`: number of nodes
        `n_edges`: number of vertices
        `n_node_features`: size of vertex features
        `n_edge_features`: size of edge features
        `n_labels`: size of labels

    Notes:

    In the additional parameters of the kvargs passed to the constructor, you can define
    additional graph attributes (they will be assigned automatically).

    Data is stored in Numpy arrays or in sparse Scipy matrices,
    labels can also be scalars.


    A graph neural network usually assumes that various data matrices
    they have certain forms, although this is not strictly observed to ensure greater flexibility.
    Basically, node attributes should have the form `(n_nodes, n_node_features)`, and the adjacency matrix
    can have the form `(n_nodes, n_nodes)`.

    Edge attributes can be stored in a dense format in the form of arrays of the form
    `(n_nodes, n_nodes, n_edge_features)` or in sparse format as arrays of the form
    `(n_edges, n_edge_features)`, so there is no need to store all zeros for missing edges.

    Labels can refer to the entire graph and have the form `(n_labels, )` or to each
    a separate node, that is, have the form `(n_nodes, n_labels)`.
    """

    def __init__(self, x=None, a=None, e=None, y=None, **kwargs):
        """
        Constructor of the main Graph class

        Args:
            `x`: `np.array`, node features in the form `(n_nodes, n_node_features)`
            `a`: `np.array` or `scipy.sparse` matrix, connectivity matrix
                  with the form `(n_nodes, n_nodes)`
            `e`: `np.array`, edge features in the form `(n_nodes, n_nodes, n_edge_features)`
                  or `(n_edges, n_edge_features)`
            `y`: `np.array`, nodes or graph labels in the form `(n_nodes, n_labels)`
                  or `(n_labels, )`
            **kwargs: additional attributes
        """
        if x is not None:
            if not isinstance(x, np.ndarray):
                raise ValueError(f"Unsupported `x` type: {type(x)}")
            if len(x.shape) == 1:
                x = x[:, None]
                warnings.warn(f"`x` will be automatically brought to the form {x.shape}")
            if len(x.shape) != 2:
                raise ValueError(
                    f"`x` must have the form `(n_nodes, n_node_features)`, set rank "
                    f"{len(x.shape)}"
                )

        if a is not None:
            if not (isinstance(a, np.ndarray) or sp.isspmatrix(a)):
                raise ValueError(f"Unsupported `a` type: {type(a)}")
            if len(a.shape) != 2:
                raise ValueError(
                    f"`a` must have a form `(n_nodes, n_nodes)`, set rank "
                    f"{len(a.shape)}"
                )

        if e is not None:
            if not isinstance(e, np.ndarray):
                raise ValueError(f"Unsupported `e` type: {type(e)}")
            if len(e.shape) not in (2, 3):
                raise ValueError(
                    f"`e` must have a form `(n_edges, n_edge_features)` or "
                    f"`(n_nodes, n_nodes, n_edge_features)`, "
                    f"set rank {len(e.shape)}"
                )

        logger.info("Read the basic properties of the graph")

        self.x = x
        self.a = a
        self.e = e
        self.y = y

        logger.info("Get additional arguments and connect them to the Graph")

        for k, v in kwargs.items():
            self[k] = v

    def numpy(self) -> tuple:
        """
        Returns all data as a tuple in the order `x`, `a`, `e`, `y`.

        Returns:

        """
        return tuple(
            main_attribute
            for main_attribute in [self.x, self.a, self.e, self.y]
            if main_attribute is not None
        )

    def get(self, *keys) -> tuple:
        """
        Returns attributes by name.

        Args:
            *keys: attribute names

        Returns:

        """
        return tuple(self[key] for key in keys if self[key] is not None)

    def __setitem__(self, key: Any, value: Any) -> Any:
        """
        Attribute Setter.

        Args:
            key: attribute key
            value: attribute value

        Returns:

        """
        setattr(self, key, value)

    def __getitem__(self, key: Any) -> Any:
        """
        Attribute Getter.

        Args:
            key: attribute key

        Returns:

        """
        return getattr(self, key, None)

    def __contains__(self, key: Any) -> bool:
        """
        Indication of the presence of an attribute by key.

        Args:
            key: attribute key

        Returns:

        """
        return key in self.keys

    def __repr__(self) -> str:
        """
        Graph representation during printing.

        Returns:

        """
        return (
            f"Graph("
            f"n_nodes={self.n_nodes}, "
            f"n_node_features={self.n_node_features}, "
            f"n_edge_features={self.n_edge_features}, "
            f"n_labels={self.n_labels})"
        )

    @property
    def n_nodes(self):
        """
        Graph number of nodes.

        Returns:

        """
        if self.x is not None:
            return self.x.shape[-2]
        elif self.a is not None:
            return self.a.shape[-1]
        else:
            return None

    @property
    def n_edges(self):
        """
        Graph number of edges.

        Returns:

        """
        if sp.issparse(self.a):
            return self.a.nnz
        elif isinstance(self.a, np.ndarray):
            return np.count_nonzero(self.a)
        else:
            return None

    @property
    def n_node_features(self):
        """
        The size of the features of the vertices of the graph.

        Returns:

        """
        if self.x is not None:
            return self.x.shape[-1]
        else:
            return None

    @property
    def n_edge_features(self):
        """
        The size of the features of the edges of the graph.

        Returns:

        """
        if self.e is not None:
            return self.e.shape[-1]
        else:
            return None

    @property
    def n_labels(self):
        """
        Size of graph labels.

        Returns:

        """
        if self.y is not None:
            shp = np.shape(self.y)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None

    @property
    def keys(self):
        """
        Graph attribute Keys.

        Returns:

        """
        keys = [
            key
            for key in self.__dict__.keys()
            if self[key] is not None and not key.startswith("__")
        ]
        return keys


def graph_fabric(x=None, a=None, e=None, y=None, **kwargs):
    return Graph(x, a, e, y, **kwargs)
