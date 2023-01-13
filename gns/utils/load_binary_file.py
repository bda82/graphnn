from os import PathLike
from typing import Any

import joblib


def load_binary_file(filename: str | bytes | PathLike[str] | PathLike[bytes] | int) -> Any:
    """
    Load binary data from file, serialized with pickle module
    
    Args:
        filename: string or path-like object

    Returns: 
        uploaded data
    """
    try:
        return joblib.load(filename)
    except ValueError:
        import pickle

        with open(filename, "rb") as f:
            return pickle.load(f, encoding="latin1")
