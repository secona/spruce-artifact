import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import io, sparse
from benchmarks.result import BenchmarkResult
from benchmarks.args import parse_run_args
from benchmarks.inputs import load_csf, load_dense

INT_MAX = 2**31 - 1
OPERATIONS = ("spmv", "sddmm", "spgemm", "mttkrp")


def main():
  args = parse_run_args(OPERATIONS)

  if args.operation is None or args.operation == "spmv":
    from benchmarks.spmv import config

    for A_filename, x_filename in config.CASES:
      A_path = Path("dataset") / A_filename
      x_path = Path("dataset") / x_filename
      run_spmv(A_path, x_path, args.repeat, args.device, args.warmup)

  if args.operation is None or args.operation == "sddmm":
    from benchmarks.sddmm import config

    for A_filename, B_filename, C_filename in config.CASES:
      A_path = Path("dataset") / A_filename
      B_path = Path("dataset") / B_filename
      C_path = Path("dataset") / C_filename
      run_sddmm(A_path, B_path, C_path, args.repeat, args.device, args.warmup)

  if args.operation is None or args.operation == "spgemm":
    from benchmarks.spgemm import config

    for A_filename, B_filename in config.CASES:
      A_path = Path("dataset") / A_filename
      B_path = Path("dataset") / B_filename
      run_spgemm(A_path, B_path, args.repeat, args.device, args.warmup)

  if args.operation is None or args.operation == "mttkrp":
    from benchmarks.mttkrp import config

    for A_filename, B_filename, C_filename in config.CASES:
      A_path = Path("dataset") / A_filename
      B_path = Path("dataset") / B_filename
      C_path = Path("dataset") / C_filename
      run_mttkrp(A_path, B_path, C_path, args.repeat, args.device, args.warmup)


def run_spmv(A_path, x_path, repeat, device, warmup):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_spmv.so"
  assert library_path.is_file()

  matrix = io.mmread(A_path, spmatrix=False)
  assert not np.iscomplexobj(matrix)

  A = sparse.csr_matrix(matrix, dtype=np.float32)
  A.sum_duplicates()
  A.sort_indices()
  m, n = A.shape

  assert min(m, n) >= 1
  assert max(m, n, A.nnz) <= INT_MAX
  assert np.all(np.isfinite(A.data))

  offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
  columns = np.ascontiguousarray(A.indices, dtype=np.int32)
  values = np.ascontiguousarray(A.data, dtype=np.float32)
  x = np.load(x_path)
  samples = np.empty(repeat, dtype=np.float32)

  library = ct.CDLL(str(library_path))
  benchmark = library.spruce_benchmark_spmv
  benchmark.restype = ct.c_char_p
  benchmark.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int64,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
  ]

  error = benchmark(
    device,
    m,
    n,
    A.nnz,
    offsets.ctypes.data,
    columns.ctypes.data,
    values.ctypes.data,
    x.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="cusparse",
    operation="spmv",
    operand_names=[A_path.name, x_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
  )
  result.print_summary()
  result.write_csv()


def run_sddmm(A_path, B_path, C_path, repeat, device, warmup):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_sddmm.so"
  assert library_path.is_file()

  matrix = io.mmread(A_path, spmatrix=False)
  assert not np.iscomplexobj(matrix)

  A = sparse.csr_matrix(matrix, dtype=np.float32)
  A.sum_duplicates()
  A.sort_indices()
  m, n = A.shape

  assert min(m, n) >= 1
  assert max(m, n, A.nnz) <= INT_MAX
  assert np.all(np.isfinite(A.data))

  factors = []
  for path, rows in ((B_path, m), (C_path, n)):
    factor = np.load(path)
    assert not np.iscomplexobj(factor)
    assert factor.ndim == 2 and factor.shape[0] == rows
    assert 1 <= factor.shape[1] <= INT_MAX
    factor = np.ascontiguousarray(factor, dtype=np.float32)
    assert np.all(np.isfinite(factor))
    factors.append(factor)
  B, C = factors
  assert B.shape[1] == C.shape[1]
  rank = B.shape[1]

  offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
  columns = np.ascontiguousarray(A.indices, dtype=np.int32)
  values = np.ascontiguousarray(A.data, dtype=np.float32)
  samples = np.empty(repeat, dtype=np.float32)

  library = ct.CDLL(str(library_path))
  benchmark = library.spruce_benchmark_sddmm
  benchmark.restype = ct.c_char_p
  benchmark.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int64,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
  ]

  error = benchmark(
    device,
    m,
    n,
    rank,
    A.nnz,
    offsets.ctypes.data,
    columns.ctypes.data,
    values.ctypes.data,
    B.ctypes.data,
    C.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="cusparse",
    operation="sddmm",
    operand_names=[A_path.name, B_path.name, C_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
  )
  result.print_summary()
  result.write_csv()


def run_spgemm(A_path, B_path, repeat, device, warmup):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_spgemm.so"
  assert library_path.is_file()

  matrices = []
  for path in (A_path, B_path):
    matrix = io.mmread(path, spmatrix=False)
    assert not np.iscomplexobj(matrix)

    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()

    assert min(matrix.shape) >= 1
    assert max(*matrix.shape, matrix.nnz) <= INT_MAX
    assert np.all(np.isfinite(matrix.data))

    matrices.append(matrix)

  A, B = matrices
  m, k = A.shape
  assert B.shape[0] == k
  n = B.shape[1]

  offsets_a = np.ascontiguousarray(A.indptr, dtype=np.int32)
  columns_a = np.ascontiguousarray(A.indices, dtype=np.int32)
  values_a = np.ascontiguousarray(A.data, dtype=np.float32)
  offsets_b = np.ascontiguousarray(B.indptr, dtype=np.int32)
  columns_b = np.ascontiguousarray(B.indices, dtype=np.int32)
  values_b = np.ascontiguousarray(B.data, dtype=np.float32)
  samples = np.empty(repeat, dtype=np.float32)

  library = ct.CDLL(str(library_path))
  benchmark = library.spruce_benchmark_spgemm
  benchmark.restype = ct.c_char_p
  benchmark.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int64,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int64,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
  ]

  error = benchmark(
    device,
    m,
    k,
    n,
    A.nnz,
    offsets_a.ctypes.data,
    columns_a.ctypes.data,
    values_a.ctypes.data,
    B.nnz,
    offsets_b.ctypes.data,
    columns_b.ctypes.data,
    values_b.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="cusparse",
    operation="spgemm",
    operand_names=[A_path.name, B_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
  )
  result.print_summary()
  result.write_csv()


def run_mttkrp(A_path, B_path, C_path, repeat, device, warmup):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_mttkrp.so"
  assert library_path.is_file()

  shape, positions_j, coordinates_j, positions_k, coordinates_k, values = load_csf(A_path)
  m, n, p = shape
  b = load_dense(B_path, n, 2)
  c = load_dense(C_path, p, 2)
  if b.shape[1] != c.shape[1]:
    raise ValueError("MTTKRP factor ranks must match")
  rank = b.shape[1]

  # Unfold mode 0 into CSR, retaining only columns for observed (j,k) pairs.
  # A full n*p by rank Khatri-Rao product can be much larger than the tensor.
  leaf_j = np.repeat(coordinates_j, np.diff(positions_k).astype(np.int64))
  pairs, columns = np.unique(
    leaf_j.astype(np.int64) * p + coordinates_k.astype(np.int64),
    return_inverse=True,
  )
  offsets = np.ascontiguousarray(positions_k[positions_j], dtype=np.int32)
  columns = np.ascontiguousarray(columns, dtype=np.int32)
  pair_j = np.ascontiguousarray(pairs // p, dtype=np.int32)
  pair_k = np.ascontiguousarray(pairs % p, dtype=np.int32)
  samples = np.empty(repeat, dtype=np.float32)

  library = ct.CDLL(str(library_path))
  benchmark = library.spruce_benchmark_mttkrp
  benchmark.restype = ct.c_char_p
  benchmark.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int64,
    ct.c_int,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
  ]

  error = benchmark(
    device,
    m,
    n,
    p,
    rank,
    values.size,
    pairs.size,
    offsets.ctypes.data,
    columns.ctypes.data,
    values.ctypes.data,
    pair_j.ctypes.data,
    pair_k.ctypes.data,
    b.ctypes.data,
    c.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="cusparse",
    operation="mttkrp",
    operand_names=[A_path.name, B_path.name, C_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
  )
  result.print_summary()
  result.write_csv()


if __name__ == "__main__":
  main()
