import os
import glob
import logging
import numpy as np
from gns.graph.graph import Graph
from gns.dataset.dataset import Dataset

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
        logger.info("Initial parameters...")
        n = self.n_count
        node_features = np.random.randint(0, self.n_features, size=n)

        # Build X array (nodes features)
        logger.info("Build and fill matrix of node features...")
        x = np.zeros((n, self.n_features))
        x[np.arange(n), node_features] = 1

        # Build A array (edges between nodes)
        logger.info("Build and fill matrix of edges...")
        a = np.random.rand(n, n) <= self.p
        a = np.maximum(a, a.T).astype(int)
        # a = sp.csr_matrix(a)

        # Build Y array (labels of nodes)
        logger.info("Build and fill matrix of labels...")
        y = np.zeros((self.n_features,))
        n_features_counts = x.sum(0)
        y[np.argmax(n_features_counts)] = 1

        # Return multidimensional array of settings
        return [x, a,y]

    def download(self):
        # Create the directory
        os.makedirs(self.path, exist_ok=True)

        # Generate required amount of samples, then save to disk
        logger.info(f"Generate {self.n_samples} graphs...")
        zeros = len(str(self.n_samples))
        for s in range(self.n_samples):
            graph = self.make_graph()
            filename = os.path.join(self.path, f'graph_{str(s).zfill(zeros)}')
            np.savez(
                filename,
                x=graph[0],
                a=graph[1],
                y=graph[2],
            )

    def read(self):
        # Read files from disk
        graphs = glob.glob(self.path + "/graph_*.npz")

        # We must return a list of Graph objects
        output = []

        # Read graphs
        for graph in graphs:
            # Read from file
            data = np.load(graph, allow_pickle=True)

            # Append arrays to Graph object
            output.append(
                Graph(
                    x=data['x'],
                    a=data['a'],
                    y=data['y']
                )
            )

        # Return array of graphs
        return output


def sfedu_dataset_fabric(n_samples=10, **kwargs):
    return SfeduDataset(n_samples=n_samples, **kwargs)
