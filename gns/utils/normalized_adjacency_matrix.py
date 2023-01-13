from gns.utils.degree_power import calculate_degree_power


def normalized_adjacency_matrix(A, symmetric=True):
    """
    Normalizes a given adjacency matrix using a degree matrix 
    
    Args:
        A: rank 2 array or sparse matrix
        symmetric: bool, a feature indicating whether to calculate symmetric normalization

    Returns:
        normalized adjacency matrix
    """
    if symmetric:
        normalized_matrix = calculate_degree_power(A, -0.5)
        result = normalized_matrix.dot(A).dot(normalized_matrix)
    else:
        normalized_matrix = calculate_degree_power(A, -1.0)
        result = normalized_matrix.dot(A)

    return result
