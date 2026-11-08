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


def load_csf(path):
    """Load a one-based 3D FROSTT tensor as CSF with a dense outer level."""
    entries = np.loadtxt(
        path,
        dtype=[("i", np.int64), ("j", np.int64), ("k", np.int64), ("value", np.float32)],
        comments=("#", "%"),
        ndmin=1,
    )
    assert entries.size >= 1
    assert entries.size <= 2**31 - 1
    shape = tuple(int(entries[name].max()) for name in ("i", "j", "k"))
    assert max(shape) <= 2**31 - 1
    for name in ("i", "j", "k"):
        assert entries[name].min() >= 1
        entries[name] -= 1
    assert np.all(np.isfinite(entries["value"]))
    entries.sort(order=("i", "j", "k"))

    # Coalesce duplicate coordinates before constructing unique CSF levels.
    starts = np.zeros(entries.size, dtype=bool)
    starts[0] = True
    for name in ("i", "j", "k"):
        starts[1:] |= entries[name][1:] != entries[name][:-1]
    indices = np.flatnonzero(starts)
    values = np.ascontiguousarray(np.add.reduceat(entries["value"], indices))
    assert np.all(np.isfinite(values))
    entries = entries[indices]

    starts = np.zeros(entries.size, dtype=bool)
    starts[0] = True
    for name in ("i", "j"):
        starts[1:] |= entries[name][1:] != entries[name][:-1]
    fibers = np.flatnonzero(starts)
    positions_j = np.zeros(shape[0] + 1, dtype=np.uint64)
    positions_j[1:] = np.cumsum(np.bincount(entries["i"][fibers], minlength=shape[0]))
    coordinates_j = np.ascontiguousarray(entries["j"][fibers], dtype=np.uint64)
    positions_k = np.empty(fibers.size + 1, dtype=np.uint64)
    positions_k[:-1] = fibers
    positions_k[-1] = entries.size
    coordinates_k = np.ascontiguousarray(entries["k"], dtype=np.uint64)
    return shape, positions_j, coordinates_j, positions_k, coordinates_k, values


def check_output(actual, reference, absolute_products):
    """Allow float32 reduction rounding, including cancellation near zero."""
    if not np.all(np.isfinite(actual)):
        raise ValueError("Benchmark output contains nonfinite values")
    np.testing.assert_array_less(
        np.abs(actual - reference), 1e-5 + 1e-4 * absolute_products,
        err_msg="Benchmark output differs from the reference",
    )
