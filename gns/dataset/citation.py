import os
import os.path as osp
import logging

import networkx as nx
import numpy as np
import requests
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from gns.config.settings import settings_fabric
from gns.dataset.dataset import Dataset
from gns.dataset.dataset_folder import DATASET_FOLDER
from gns.graph.graph import Graph
from gns.utils.idx_to_mask import get_mask_by_indexes
from gns.utils.preprocess_features import preprocess_features
from gns.utils.read_file import read_file

settings = settings_fabric()

logger = logging.getLogger(__name__)


class Citation(Dataset):
    """
    Data sets for citation dataset:
        - Cora
        - Citeseer
        - Pubmed.

    Node attributes are vectors of a set of words representing
    the most common words in a text document,
    associated with each node.

    For example, two object nodes are connected to each other if
    one of them quotes the other.

    The train, test and validation sections are set as binary masks and
    are accessible via the `mask_tr`, `mask_va` and `mask_te` attributes.

    Input parameters:
        name: symbolic name of the dataset (`cora", `citeseer", or `pubmed")
        random_split: if True, returns a random division of the dataset into subsets
            (20 nodes per class for training, 30 nodes per class for validation and N remaining nodes for testing).
        normalize_x: if True, the features are normalized
        dtype: the `dtype` parameter from the numpy module for graph data.
    """

    citation_url = settings.urls.citation_url
    citation_suffixes = settings.constants.citation_suffixes

    def __init__(
        self,
        name: str,
        random_split: bool = False,
        normalize_x: bool = False,
        dtype: np.core.single = np.float32,
        **kwargs,
    ):
        if hasattr(dtype, settings.data_types.as_numpy_dtype):
            dtype = dtype.as_numpy_dtype

        if name.lower() not in self.available_datasets():
            raise ValueError(
                f"Unknown dataset {name}. Available datasets save in "
                f"list {self.__class__.__name__}.available_datasets()"
            )

        self.name = name.lower()
        self.random_split = random_split
        self.normalize_x = normalize_x
        self.mask_tr = self.mask_va = self.mask_te = None
        self.dtype = dtype

        super().__init__(**kwargs)

    @property
    def path(self) -> str:
        return osp.join(
            DATASET_FOLDER,
            settings.folders.citation,
            self.name
        )

    def read(self) -> list[Graph]:
        logger.info(f"Get cora objects.")
        objects = [read_file(self.path, self.name, s) for s in self.citation_suffixes]
        objects = [o.A if sp.issparse(o) else o for o in objects]

        # open objects into variables

        x, y, tx, ty, allx, ally, graph, idx_te = objects

        idx_tr = np.arange(y.shape[0])
        idx_va = np.arange(y.shape[0], y.shape[0] + 500)
        idx_te = idx_te.astype(int)
        idx_te_sort = np.sort(idx_te)

        logger.info(f"Fixing disabled nodes for the model {self.name}.")

        if self.name == settings.names.citeseer:
            idx_te_len = idx_te.max() - idx_te.min() + 1
            tx_ext = np.zeros((idx_te_len, x.shape[1]))
            tx_ext[idx_te_sort - idx_te.min(), :] = tx
            tx = tx_ext
            ty_ext = np.zeros((idx_te_len, y.shape[1]))
            ty_ext[idx_te_sort - idx_te.min(), :] = ty
            ty = ty_ext

        x = np.vstack((allx, tx))
        y = np.vstack((ally, ty))

        x[idx_te, :] = x[idx_te_sort, :]
        y[idx_te, :] = y[idx_te_sort, :]

        logger.info(f"If Normalization of features enabled: {self.normalize_x}.")

        if self.normalize_x:
            logger.info("Preparation of node features.")
            x = preprocess_features(x)

        logger.info(f"If Random separation enabled: {self.random_split}.")

        if self.random_split:
            logger.info("Preparing a random split.")
            y_indices = np.arange(y.shape[0])
            y_class_number = y.shape[1]
            idx_tr, idx_te, _, y_te = train_test_split(
                y_indices,
                y,
                train_size=20 * y_class_number,
                stratify=y
            )
            idx_va, idx_te = train_test_split(
                idx_te,
                train_size=30 * y_class_number,
                stratify=y_te
            )

        logger.info("Adjacency matrix format.")

        adjacency_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adjacency_matrix.setdiag(0)
        adjacency_matrix.eliminate_zeros()

        logger.info(f"Creating masks for training/validation/testing.")

        self.mask_tr = get_mask_by_indexes(idx_tr, y.shape[0])
        self.mask_va = get_mask_by_indexes(idx_va, y.shape[0])
        self.mask_te = get_mask_by_indexes(idx_te, y.shape[0])

        return [
            Graph(
                x=x.astype(self.dtype),
                a=adjacency_matrix.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]

    def download(self) -> None:
        logger.info(f"Uploading a dataset {self.name}...")

        os.makedirs(self.path, exist_ok=True)

        for n in self.citation_suffixes:
            f_name = f"{settings.datasets.dataset_prefix_index}{self.name}.{n}"
            try:
                req = requests.get(self.citation_url.format(f_name))
            except Exception as ex:
                logger.error(f"Dataset server error: {ex}")
                raise ex

            if req.status_code == 404:
                raise ValueError(
                    f"Unable to load dataset {self.citation_url.format(f_name)} (Seems that server status is 404)."
                )

            with open(os.path.join(self.path, f_name), "wb") as out_file:
                out_file.write(req.content)

        logger.info(f"Uploading a dataset {self.name} completed.")

    @staticmethod
    def available_datasets():
        """
        Get available datasets.
        """
        return settings.datasets.available_datasets


def citation_fabric(
    name,
    random_split: bool = False,
    normalize_x: bool = False,
    dtype: np.core.single = np.float32,
    **kwargs,
):
    return Citation(name, random_split, normalize_x, dtype, **kwargs)
