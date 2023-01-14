import os
import glob
import logging
import numpy as np
import scipy.sparse as sp
import json
import requests
from pathlib import Path
from types import SimpleNamespace
from gns.graph.graph import Graph
from gns.dataset.dataset import Dataset
from gns.transformation.normalized_adjacency import NormalizeAdjacencyMatrix
from gns.utils.read_file import read_file
from gns.utils.load_binary_file import load_binary_file
from gns.utils.idx_to_mask import get_mask_by_indexes
from gns.utils.preprocess_features import preprocess_features
from gns.dataset.dataset_folder import DATASET_FOLDER

logger = logging.getLogger(__name__)


class TechDataset(Dataset):
    """
    Dataset with large amount of nodes, for representing all IT technologies.
    """

    dataset_url = 'https://raw.githubusercontent.com/ZhongTr0n/JD_Analysis/main/'
    dataset_suffix = 'json'

    def __init__(
            self,
            name,
            random_split: bool = False,
            normalize_x: bool = False,
            dtype: np.core.single = np.float32,
            **kwargs,
    ):
        if name.lower() not in self.available_datasets():
            raise ValueError(
                f"Unknown dataset {name}. Available datasets save in "
                f"list {self.__class__.__name__}.available_datasets()"
            )

        self.name = name
        self.random_split = random_split
        self.normalize_x = normalize_x
        self.mask_tr = self.mask_va = self.mask_te = None
        self.dtype = dtype

        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, self.__class__.__name__)

    @staticmethod
    def available_datasets():
        return ["jd_data", "jd_data2"]

    def download(self):
        logger.info(f"Downloading a dataset {self.name}...")

        os.makedirs(self.path, exist_ok=True)

        dataset_full_name = self.name + '.' + self.dataset_suffix
        dataset_full_url = self.dataset_url + dataset_full_name

        try:
            req = requests.get(dataset_full_url)
        except Exception as ex:
            logger.error(f"Dataset server error: {ex}")
            raise ex

        if req.status_code == 404:
            raise ValueError(
                f"Unable to load dataset {dataset_full_url} (Seems that server status is 404)."
            )

        with open(os.path.join(self.path, dataset_full_name), "wb") as out_file:
            out_file.write(req.content)

        # Load file and parse JSON
        file = self.path + '/' + self.name
        fileJson = file + '.' + self.dataset_suffix
        f = open(fileJson, "rb")
        data = json.loads(f.read())
        f.close()

        # Read data from file
        nodes = data['nodes']
        edges = data['links']

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
            x[idxNode][0] = node['color']

        # Build A array (edges between nodes)
        a = np.zeros((n_count, n_count))
        for idxEdge, edge in enumerate(edges):
            # Detect index of node by name
            for idxNode, node in enumerate(nodes):
                if node["id"] == edge['source']:
                    sourceId = idxNode
                if node["id"] == edge['target']:
                    targetId = idxNode
            # Fill array with found data (only uniq matches)
            if a[sourceId][targetId] == 0:
                a[sourceId][targetId] = 1

        # Build E array (edges features)
        e = np.zeros((e_count, e_features))
        for idxEdge, edge in enumerate(edges):
            # Let's use value of edges as weight
            e[idxEdge][0] = edge['value']

        # Build Y array (labels of nodes)
        y = np.zeros((n_count, n_labels))
        for idxNode, node in enumerate(nodes):
            # Index of node, in future here should be IDs of nodes from DB
            y[idxNode][0] = idxNode

        # Save archive
        filename = os.path.join(self.path, file)
        np.savez(filename, x=x, a=a, e=e, y=y)

        logger.info(f"Uploading a dataset {dataset_full_name} completed.")

    def read(self) -> list[Graph]:
        logger.info(f"Get {self.name} objects.")

        # Download dataset if not exists
        file = self.path + '/' + self.name + '.npz'
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

        logger.info(f"If Normalization of features enabled: {self.normalize_x}.")

        if self.normalize_x:
            logger.info("Preparation of node features.")
            x = preprocess_features(x)

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


def tech_dataset_fabric(
        name,
        random_split: bool = False,
        normalize_x: bool = False,
        dtype: np.core.single = np.float32,
        **kwargs
):
    return TechDataset(name, random_split, normalize_x, dtype, **kwargs)
