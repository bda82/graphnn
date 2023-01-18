import os
import logging
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from gns.graph.graph import Graph
from gns.dataset.dataset import Dataset
from gns.dataset.dataset_folder import DATASET_FOLDER
from gns.utils.idx_to_mask import get_mask_by_indexes
from arango import ArangoClient

logger = logging.getLogger(__name__)


class ArangoTechDataset(Dataset):
    def __init__(self, dtype: np.core.single = np.float32, **kwargs):
        self.dtype = dtype
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, self.__class__.__name__)

    def download(self):
        logger.info(f"Downloading a dataset from ArangoDB...")

        # Create the directory
        os.makedirs(self.path, exist_ok=True)

        # Initialize the ArangoDB client.
        client = ArangoClient()

        # Connect to "_system" database as root user.
        db = client.db('_system', username='root', password='root_pass')

        # Get list of nodes
        nodes_collection = db.collection('nodes')
        nodes = nodes_collection.all()
        nodes = np.fromiter(nodes, dtype=np.dtype(dict))

        # Get list of edges
        edges_collection = db.collection('edges')
        edges = edges_collection.all()
        edges = np.fromiter(edges, dtype=np.dtype(dict))

        # Some initial values
        n_features = 1
        n_count = len(nodes)
        n_labels = 1
        e_features = 1
        e_count = len(edges)

        # Build X array (nodes features)
        x = np.zeros((n_count, n_features))
        for idxNode, node in enumerate(nodes):
            # Let's use color of nodes
            x[idxNode][0] = 1

        # Build A array (edges between nodes)
        a = np.zeros((n_count, n_count))
        for idxEdge, edge in enumerate(edges):
            # Detect index of node by name
            for idxNode, node in enumerate(nodes):
                if node["_id"] == edge['_from']:
                    sourceId = idxNode
                if node["_id"] == edge['_to']:
                    targetId = idxNode
            # Fill array with found data (only uniq matches)
            if a[sourceId][targetId] == 0:
                a[sourceId][targetId] = 1

        # Build E array (edges features)
        e = np.zeros((e_count, e_features))
        for idxEdge, edge in enumerate(edges):
            # Let's use value of edges as weight
            e[idxEdge][0] = 1

        # Build Y array (labels of nodes)
        y = np.zeros((n_count, n_labels))
        for idxNode, node in enumerate(nodes):
            # Index of node, in future here should be IDs of nodes from DB
            y[idxNode][0] = node['_key']

        # Save archive
        file = self.path + '/graph.npz'
        np.savez(file, x=x, a=a, e=e, y=y)

        logger.info(f"Uploading a dataset {file} completed.")

    def read(self) -> list[Graph]:
        logger.info(f"Get objects...")

        # Download dataset if not exists
        file = self.path + '/graph.npz'
        file_path = Path(file)
        if not file_path.is_file():
            self.download()

        # Load file
        data = np.load(file, allow_pickle=True)

        # Params
        x = data['x']
        y = data['y']
        e = data['e']
        a = sp.csr_matrix(data['a'])

        # Indexes for creating masks
        idx_tr = np.arange(0, int(y.shape[0] / 8))
        idx_te = np.arange(int(y.shape[0] / 8), y.shape[0])
        idx_va = np.arange(int(y.shape[0] / 4), int(y.shape[0] / 2))

        logger.info(f"Creating masks for training/validation/testing.")

        self.mask_tr = get_mask_by_indexes(idx_tr, y.shape[0])
        self.mask_te = get_mask_by_indexes(idx_te, y.shape[0])
        self.mask_va = get_mask_by_indexes(idx_va, y.shape[0])

        # Return final graph
        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
                e=e.astype(self.dtype),
            )
        ]


def arango_tech_dataset_fabric(dtype: np.core.single = np.float32, **kwargs):
    return ArangoTechDataset(dtype=dtype, **kwargs)
