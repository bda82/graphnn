import tensorflow as tf

from tensorflow import IndexedSlices, SparseTensor, Tensor
from tensorflow.keras import backend as KerasBackend  # noqa
from tensorflow.python.ops.linalg.sparse import sparse as tensorflow_sparse

from gns.utils.transpose import transpose


def dot_production(a, b) -> object | Tensor | None | IndexedSlices | SparseTensor:
    """
    Calculates the multiplication of `a @ b` for a and b of the same rank (both 2 or both 3 ranks).

    If the rank is 2, then the innermost dimension `a` should correspond to the
    external dimension `b`.

    It can be some cases:
    1) If the rank is 3, then the first dimension of `a` and `b` must be even, and
       the function calculates matrix multiplication.
    2) In these cases, we can re-use tensorflow_sparse.CSRSparseMatrix
       rank 2 (sparse-sparse), rank 3 (sparse-dense), rank 3 (dense-sparse), rank 3 (sparse-sparse)
    3) In this case we just use standard mult.
       rank 2 (dense-dense), rank 3 (dense-dense)

    Supports both dense and sparse multiplications for matrices and arrays.

        # First case: rank 2 sparse, rank 2 sparse (sparse-dense)
    # In these cases, we can use the faster sparse-dense multiplication method from tensotflow.sparse


    Args:
        a: tensor or SparseTensor with rank 2 or 3
        b: tensor or SparseTensor with the same rank 2 or 3
    
    Returns:
        tensor or SparseTensor with rank 2 or 3
    """
    dimension_a = KerasBackend.ndim(a)
    dimension_b = KerasBackend.ndim(b)

    if dimension_a != dimension_b:
        raise ValueError(f"Expected equival rank, but pass rank {dimension_a} and {dimension_b}")

    issparse_a_predicate = KerasBackend.is_sparse(a)
    issparse_b_predicate = KerasBackend.is_sparse(b)

    # The first case

    if dimension_a == 2:
        if issparse_a_predicate and not issparse_b_predicate:
            return tf.sparse.sparse_dense_matmul(a, b)
        if not issparse_a_predicate and issparse_b_predicate:
            return transpose(tf.sparse.sparse_dense_matmul(transpose(b), transpose(a)))

    # The second case

    if issparse_a_predicate:
        a = tensorflow_sparse.CSRSparseMatrix(a)

    if issparse_b_predicate:
        b = tensorflow_sparse.CSRSparseMatrix(b)

    if issparse_a_predicate or issparse_b_predicate:
        out = tensorflow_sparse.matmul(a, b)
        if hasattr(out, "to_sparse_tensor"):
            return out.to_sparse_tensor()
        else:
            return out

    # The third case
    
    return tf.matmul(a, b)
