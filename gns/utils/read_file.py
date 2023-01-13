import os

import numpy as np

from gns.utils.load_binary_file import load_binary_file
from gns.config.settings import settings_fabric

settings = settings_fabric()


def read_file(path, name, suffix):
    """
    Read file with data.

    Args:
        path: file path
        name: file name
        suffix: file suffix

    Returns:
        file content
    """
    full_filename = os.path.join(
        path, f"{settings.datasets.dataset_prefix_index}{name}.{suffix}"
    )
    
    if suffix == "test.index":
        return np.loadtxt(full_filename)

    return load_binary_file(full_filename)
