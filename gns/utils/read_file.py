import os

import numpy as np

from gns.utils.load_binary import load_binary


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
    full_fname = os.path.join(path, "ind.{}.{}".format(name, suffix))
    
    if suffix == "test.index":
        return np.loadtxt(full_fname)

    return load_binary(full_fname)
