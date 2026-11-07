"""Validated float32 inputs shared by benchmark runners."""

import numpy as np
from scipy import io, sparse


def load_csr(path, index_dtype=np.int32):
    matrix = io.mmread(path, spmatrix=False)
    if np.iscomplexobj(matrix):
        raise ValueError(f"Complex matrices are unsupported: {path}")
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if min(matrix.shape) < 1 or max(*matrix.shape, matrix.nnz) > 2**31 - 1:
        raise ValueError(f"Dimensions must be positive and dimensions/nnz must fit int32: {path}")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"Matrix values must be finite float32 numbers: {path}")
    matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=index_dtype)
    matrix.indices = np.ascontiguousarray(matrix.indices, dtype=index_dtype)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=np.float32)
    return matrix


def load_dense(path, rows, ndim):
    array = np.load(path)
    if np.iscomplexobj(array) or array.ndim != ndim or array.shape[0] != rows:
        raise ValueError(f"Expected a real {ndim}D array with {rows} rows: {path}")
    if min(array.shape) < 1 or max(array.shape) > 2**31 - 1:
        raise ValueError(f"Array dimensions must be positive and fit int32: {path}")
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Array values must be finite float32 numbers: {path}")
    return array


def check_output(actual, reference, absolute_products):
    """Allow float32 reduction rounding, including cancellation near zero."""
    if not np.all(np.isfinite(actual)):
        raise ValueError("Benchmark output contains nonfinite values")
    np.testing.assert_array_less(
        np.abs(actual - reference), 1e-5 + 1e-4 * absolute_products,
        err_msg="Benchmark output differs from the reference",
    )
