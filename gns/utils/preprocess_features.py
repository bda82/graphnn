import numpy as np
import scipy.sparse as sp


def preprocess_features(features):
    """
    Process features.

    Args:
        features: features
    
    Returns:
        features
    """
    row_summa = np.array(features.sum(1))

    r_inv = np.power(row_summa, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.0

    r_mat_inv = sp.diags(r_inv)

    return r_mat_inv.dot(features)
