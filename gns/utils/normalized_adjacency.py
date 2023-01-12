from gns.utils.degree_power import degree_power


def normalized_adjacency(A, symmetric=True):
    """
    Normalizes a given adjacency matrix using a degree matrix 
    
    Args:
        A: rank 2 array or sparse matrix
        symmetric: bool, a feature indicating whether to calculate symmetric normalization

    Returns:
        normalized adjacency matrix
    """
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)
