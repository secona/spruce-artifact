import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import io, sparse
from benchmarks.result import BenchmarkResult
from benchmarks.args import parse_run_args

INT_MAX = 2**31 - 1
OPERATIONS = ("spmv", "sddmm", "spgemm")


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


if __name__ == "__main__":
  main()
