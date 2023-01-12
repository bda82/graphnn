import tensorflow as tf
from tensorflow import IndexedSlices, SparseTensor, Tensor
from tensorflow.keras import backend as K  # noqa
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

from gns.utils.transpose import transpose


def dot(a, b) -> object | Tensor | None | IndexedSlices | SparseTensor:
    """
    Calculates the multiplication of `a @ b` for a and b of the same rank (both 2 or both 3 ranks).

    If the rank is 2, then the innermost dimension `a` should correspond to the
    external dimension `b'.
    
    If the rank is 3, then the first dimension of `a` and `b` must be even, and
    the function calculates matrix multiplication.

    Supports both dense and sparse multiplications for matrices and arrays.

    Args:
        a: tensor or SparseTensor with rank 2 or 3
        b: tensor or SparseTensor with the same rank 2 or 3
    
    Returns:
        tensor or SparseTensor with rank 2 or 3
    """
    a_ndim = K.ndim(a)
    b_ndim = K.ndim(b)

    if a_ndim != b_ndim:
        raise ValueError(f"Expected equival rank, but pass rank {a_ndim} and {b_ndim}")

    a_is_sparse = K.is_sparse(a)
    b_is_sparse = K.is_sparse(b)

    # First case: rank 2 sparse, rank 2 sparse (sparse-dense)
    # In these cases, we can use the faster sparse-dense multiplication method from tensotflow.sparse

    if a_ndim == 2:
        if a_is_sparse and not b_is_sparse:
            return tf.sparse.sparse_dense_matmul(a, b)
        if not a_is_sparse and b_is_sparse:
            return transpose(tf.sparse.sparse_dense_matmul(transpose(b), transpose(a)))

    # The second case:
    # - rank 2 (sparse-sparse)
    # - rank 3 (sparse-dense),
    # - rank 3 (dense-sparse)
    # - rank 3 (sparse-sparse)
    # In these cases, we can re-use tfsp.CSRSparseMatrix

    if a_is_sparse:
        a = tfsp.CSRSparseMatrix(a)

    if b_is_sparse:
        b = tfsp.CSRSparseMatrix(b)

    if a_is_sparse or b_is_sparse:
        out = tfsp.matmul(a, b)
        if hasattr(out, "to_sparse_tensor"):
            return out.to_sparse_tensor()
        else:
            return out

    # The third case:
    # - # - rank 2 (dense-dense)
    # - rank 3 (dense-dense)
    # This is a standard piece
    
    return tf.matmul(a, b)
