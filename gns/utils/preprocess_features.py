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
    rowsum = np.array(features.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)

    features = r_mat_inv.dot(features)

    return features
