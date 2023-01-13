import os
import glob
import logging
import numpy as np
from gns.graph.graph import Graph
from gns.dataset.dataset import Dataset
from gns.transformation.normalized_adjacency import NormalizeAdjacencyMatrix

logger = logging.getLogger(__name__)


class SfeduDataset(Dataset):
    """
    Example for competence Dataset.
    """
    def __init__(
        self,
        n_samples=100,
        n_count=20,
        n_features=3,
        e_features=3,
        p=0.1,
        **kwargs
    ):
        # Amount of graphs

        self.n_samples = n_samples

        # Count of nodes in each graph

        self.n_count = n_count

        # Amount of features in each node

        self.n_features = n_features

        # Amount of features in each edge

        self.e_features = e_features

        # Probability of creating connection between nodes

        self.p = p

        super().__init__(**kwargs)

    # @property
    # def path(self):
    #     return os.path.dirname(__file__) + '/' + self.__class__.__name__

    def make_graph(self):
        # Initial parameters

        logger.info("Initial parameters...")

        n = self.n_count
        node_features = np.random.randint(0, self.n_features, size=n)

        # Build and fill matrix of node features

        logger.info("Build and fill matrix of node features...")

        x = np.zeros((n, self.n_features))
        x[np.arange(n), node_features] = 1

        # Build and fill matrix of edges, then convert to Coordinates array

        logger.info("Build and fill matrix of edges, then convert to Coordinates array...")

        a = np.random.rand(n, n) <= self.p
        a = np.maximum(a, a).astype(int)

        # Labels

        logger.info("Build labels...")

        y = np.zeros((self.n_features,))
        n_features_counts = x.sum(0)
        y[np.argmax(n_features_counts)] = 1

        # Return multidimensional array of settings

        return [x, a, y]

    def download(self):
        # Create the directory

        os.makedirs(self.path, exist_ok=True)

        # Generate required amount of samples, then save to disk

        for s in range(self.n_samples):
            graph = self.make_graph()
            filename = os.path.join(self.path, f'graph_{str(s).zfill(3)}')
            np.savez(
                filename,
                x=graph[0],
                a=graph[1],
                y=graph[2],
            )

    def read(self):
        # Read files from disk

        _graphs = glob.glob(self.path + "/graph_*.npz")

        # Output array

        graphs = []

        # Read graphs

        for graph in _graphs:
            # Read from file
            data = np.load(graph, allow_pickle=True)

            # Process parameters

            x = data['x']
            a = data['a']

            #a = sp.csr_matrix(a)

            y = data['y']

            # Append arrays to Graph object

            graphs.append(Graph(x=x, a=a, y=y))

        # Return array of graphs
        return graphs


def sfedu_dataset_fabric(n_samples=100, **kwargs):
    return SfeduDataset(n_samples=n_samples, transforms=NormalizeAdjacencyMatrix(), **kwargs)
