import copy
import os.path as osp
import warnings

import numpy as np
import tensorflow as tf

from gns.dataset.dataset_folder import DATASET_FOLDER
from gns.graph.graph import Graph
from gns.utils.get_spec import get_spec


class Dataset:
    """
    The main container for Graphs.

    It is the basic one for compiling datasets.

    To create a dataset, you need to use the read() method, which should return list of Graph objects.

    ```py
    class My Custom Data set(Dataset):
        def read(self):
            return [Graph(x=x, adj=adj, y=y) for x, adj, y in some_list]
    ```

    The download() method will be called automatically if the path defined by the variable path
    does not exist.

    The default path contains: ~/gns/datasets/ClassName/

    In this case, the dataset will be loaded first, and then read.

    Datasets should have a type and behavior compatible with Numpy arrays that use simple 1D indexing.

    Examples:
    ```py
    >>> dataset[0]
    Graph(...)

    >>> dataset[[1, 2, 3]]
    Dataset(n_graphs=3)

    >>> dataset[1:10]
    Dataset(n_graphs=9)

    >>> np.random.shuffle(dataset)

    >>> for graph in dataset[:3]:
    >>>     logger.info(graph)
    Graph(...)
    Graph(...)
    Graph(...)
    ```

    Datasets should have the following properties, which will be automatically calculated:
        `n_nodes`: the number of nodes in the dataset (always None except for datasets with single or mixed mode);
        `n_node_features`: the size of the node features, (it is assumed that they are equal for all graphs);
        `n_edge_features': size of edge objects (assumed to be equal for all graphs);
        `n_labels`: the size of the labels (it is assumed that it is the same for all graphs);
            this parameter is calculated as 'y.shape[-1]`.

    Any additional `kwargs' passed to the constructor will be automatically assigned
    as attributes of the dataset instance.

    Datasets also offer three basic manipulation functions for applying callable objects to his count:

    `apply(transform)`: replaces each graph with the output of the transform(graph) function.

    For example: function `apply(gns.transformation.NormalizeAdj())` normalizes the adjacency matrix for each dataset graph

    Function `map(transform, reduce=None)`: returns a list containing the output of the `transform(graph)` function
    for each graph. The `reduce` method is a function (`callable`) that will return `reduce(output_list)`.

    For example: function `map(lambda: g.n_nodes, reduce=np.mean)` will return the average number of nodes in the dataset.

    Functino `filter(function)`: returns from the dataset any graph for which `function(graph) is False'.

    For example: `filter(lambda: g.n_nodes < 100)` removes all graphs larger than 100 nodes from the dataset.

    For data sets in mixed mode (one adjacency matrix, multiple instances of node objects),
    they are expected to have a specific structure.

    Graphs returned by `read()` must not have adjacency matrices, and should instead be saved as a singleton
    in the datasets `a` attribute.

    For example:
    ```py
    class MyMixedModeDataset(Dataset):
        def read(self):
            self.a = compute_adjacency_matrix()
            return [Graph(x=x, y=y) for x, y in some_magic_list]
    ```

    Input parameters:
        `transforms': a callable object or a list of callable objects that are automatically
            applies to graphs after loading the dataset.
    """

    def __init__(self, transforms=None, **kwargs):
        # Used for mixed mode datasets

        self.a = None

        # Reading additional parameters

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Upload data

        if not osp.exists(self.path):
            self.download()

        # Get graph

        self.graphs = self.read()
        if self.a is None and self.__len__() > 0 and "a" not in self.graphs[0]:
            warnings.warn(
                "The graph in this dataset does not have an adjacency matrix"
            )

        # Application of transformations

        if transforms is not None:
            if not isinstance(transforms, (list, tuple)) and callable(transforms):
                transforms = [transforms]
            elif not all([callable(t) for t in transforms]):
                raise ValueError(
                    "`transforms` must be a callable"
                )
            else:
                pass
            for t in transforms:
                self.apply(t)

    def read(self):
        raise NotImplementedError

    def download(self):
        pass

    def apply(self, transform):
        if not callable(transform):
            raise ValueError("`transform` must be a callable")

        for i in range(len(self.graphs)):
            self.graphs[i] = transform(self.graphs[i])

    def map(self, transform, reduce=None):
        if not callable(transform):
            raise ValueError("`transform` must be a callable")
        if reduce is not None and not callable(reduce):
            raise ValueError("`reduce` must be a callable")

        out = [transform(g) for g in self.graphs]
        return reduce(out) if reduce is not None else out

    def filter(self, function):
        if not callable(function):
            raise ValueError("`function` must be a callable")
        self.graphs = [g for g in self.graphs if function(g)]

    def __getitem__(self, key):
        if not (
            np.issubdtype(type(key), np.integer)
            or isinstance(key, (slice, list, tuple, np.ndarray))
        ):
            raise ValueError(f"Unsupported type for key {type(key)}")
        if np.issubdtype(type(key), np.integer):
            return self.graphs[int(key)]
        else:
            dataset = copy.copy(self)
            if isinstance(key, slice):
                dataset.graphs = self.graphs[key]
            else:
                dataset.graphs = [self.graphs[i] for i in key]
            return dataset

    def __setitem__(self, key, value):
        is_iterable = isinstance(value, (list, tuple))
        if not isinstance(value, (Graph, list, tuple)):
            raise ValueError(
                "Only graphs or sequences of graphs can be assigned to datasets"
            )
        if is_iterable and not all([isinstance(v, Graph) for v in value]):
            raise ValueError(
                "The assigned sequence must contain only graphs"
            )
        if is_iterable and isinstance(key, int):
            raise ValueError(
                "It is not possible to assign multiple graphs to the same key"
            )
        if not is_iterable and isinstance(key, (slice, list, tuple)):
            raise ValueError(
                "It is not possible to assign one graph to multiple keys"
            )
        if not (isinstance(key, (int, slice, list, tuple))):
            raise ValueError(f"Unsupported key type {type(key)}")

        if isinstance(key, int):
            self.graphs[key] = value
        else:
            if isinstance(key, slice):
                self.graphs[key] = value
            else:
                for i, k in enumerate(key):
                    self.graphs[k] = value[i]

    def __add__(self, other):
        self_new = copy.copy(self)
        self_new.graphs = self.graphs + other.graphs

        return self_new

    def __len__(self):
        return len(self.graphs)

    def __repr__(self):
        return "{}(n_graphs={})".format(self.__class__.__name__, self.n_graphs)

    @property
    def path(self):
        return osp.join(DATASET_FOLDER, self.__class__.__name__)

    @property
    def n_graphs(self):
        return self.__len__()

    @property
    def n_nodes(self):
        if len(self.graphs) == 1:
            return self.graphs[0].n_nodes
        elif self.a is not None:
            return self.a.shape[-1]
        else:
            return None

    @property
    def n_node_features(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_node_features
        else:
            return None

    @property
    def n_edge_features(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_edge_features
        else:
            return None

    @property
    def n_labels(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_labels
        else:
            return None

    @property
    def signature(self):
        """
        This property calculates the signature of the dataset, which can be
        passed to `gnu.utils.to_tf_signature(signature)` for calculation
        tensor flow signature. We can ignore this property if
        a custom `GenericLoader' is created.

        The signature consists of TensorFlow TypeSpec, form and type
        all characteristic matrices of graphs in the dataset. Signature
        it is returned as a dictionary of dictionaries with the keys `x`, `a`, `e' and
        'y` for the four main data matrices.

        Each nested dictionary will have the keys `spec`, `shape` and `dtype'.

        Returns:

        """
        if len(self.graphs) == 0:
            return None
        signature = {}
        graph = self.graphs[0]  # This is always non-empty
        if graph.x is not None:
            signature["x"] = dict()
            signature["x"]["spec"] = get_spec(graph.x)
            signature["x"]["shape"] = (None, self.n_node_features)
            signature["x"]["dtype"] = tf.as_dtype(graph.x.dtype)
        if graph.a is not None:
            signature["a"] = dict()
            signature["a"]["spec"] = get_spec(graph.a)
            signature["a"]["shape"] = (None, None)
            signature["a"]["dtype"] = tf.as_dtype(graph.a.dtype)
        if graph.e is not None:
            signature["e"] = dict()
            signature["e"]["spec"] = get_spec(graph.e)
            signature["e"]["shape"] = (None, self.n_edge_features)
            signature["e"]["dtype"] = tf.as_dtype(graph.e.dtype)
        if graph.y is not None:
            signature["y"] = dict()
            signature["y"]["spec"] = get_spec(graph.y)
            signature["y"]["shape"] = (self.n_labels,)
            signature["y"]["dtype"] = tf.as_dtype(np.array(graph.y).dtype)
        return signature


def dataset_fabric(transforms=None, **kwargs):
    return Dataset(transforms, **kwargs)
