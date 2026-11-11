"""Run cuSPARSE benchmarks, or return reference arrays without timing or CSVs."""

import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from benchmarks.result import BenchmarkResult
from benchmarks.args import parse_run_args
from benchmarks.inputs import load_csf, load_csr, load_dense

OPERATIONS = ("spmv", "sddmm", "spgemm", "mttkrp")


class SpgemmOutput(ct.Structure):
  _fields_ = [
    ("nnz", ct.c_int64),
    ("offsets", ct.POINTER(ct.c_int)),
    ("columns", ct.POINTER(ct.c_int)),
    ("values", ct.POINTER(ct.c_float)),
  ]


def _csr_arrays(matrix, suffix=""):
  return {
    f"shape{suffix}": np.asarray(matrix.shape, dtype=np.int64),
    f"indptr{suffix}": matrix.indptr,
    f"indices{suffix}": matrix.indices,
    f"data{suffix}": matrix.data,
  }


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


def run_spmv(A_path, x_path, repeat, device, warmup, *, reference=False):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_spmv.so"
  assert library_path.is_file()

  A = load_csr(A_path)
  m, n = A.shape

  offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
  columns = np.ascontiguousarray(A.indices, dtype=np.int32)
  values = np.ascontiguousarray(A.data, dtype=np.float32)
  x = load_dense(x_path, n, 1)
  samples = None if reference else np.empty(repeat, dtype=np.float32)
  output = np.empty(m, dtype=np.float32) if reference else None

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
    samples.ctypes.data if samples is not None else None,
    output.ctypes.data if output is not None else None,
  )
  if error:
    raise RuntimeError(error.decode())

  if reference:
    return {**_csr_arrays(A), "x": x, "y": output}

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


def run_sddmm(A_path, B_path, C_path, repeat, device, warmup, *, reference=False):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_sddmm.so"
  assert library_path.is_file()

  A = load_csr(A_path)
  m, n = A.shape

  B = load_dense(B_path, m, 2)
  C = load_dense(C_path, n, 2)
  if B.shape[1] != C.shape[1]:
    raise ValueError("SDDMM factor ranks must match")
  rank = B.shape[1]

  offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
  columns = np.ascontiguousarray(A.indices, dtype=np.int32)
  values = np.ascontiguousarray(A.data, dtype=np.float32)
  samples = None if reference else np.empty(repeat, dtype=np.float32)
  output = np.empty(A.nnz, dtype=np.float32) if reference else None

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
    samples.ctypes.data if samples is not None else None,
    output.ctypes.data if output is not None else None,
  )
  if error:
    raise RuntimeError(error.decode())

  if reference:
    return {**_csr_arrays(A), "b": B, "c": C, "d": output}

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


def run_spgemm(A_path, B_path, repeat, device, warmup, *, reference=False):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "cusparse" / "libcusparse_spgemm.so"
  assert library_path.is_file()

  A = load_csr(A_path)
  B = load_csr(B_path)
  m, k = A.shape
  assert B.shape[0] == k
  n = B.shape[1]

  offsets_a = np.ascontiguousarray(A.indptr, dtype=np.int32)
  columns_a = np.ascontiguousarray(A.indices, dtype=np.int32)
  values_a = np.ascontiguousarray(A.data, dtype=np.float32)
  offsets_b = np.ascontiguousarray(B.indptr, dtype=np.int32)
  columns_b = np.ascontiguousarray(B.indices, dtype=np.int32)
  values_b = np.ascontiguousarray(B.data, dtype=np.float32)
  samples = None if reference else np.empty(repeat, dtype=np.float32)
  output = SpgemmOutput() if reference else None

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
    ct.POINTER(SpgemmOutput),
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
    samples.ctypes.data if samples is not None else None,
    ct.byref(output) if output is not None else None,
  )
  if error:
    raise RuntimeError(error.decode())

  if reference:
    free_output = library.spruce_free_spgemm_output
    free_output.argtypes = [ct.POINTER(SpgemmOutput)]
    free_output.restype = None
    try:
      arrays = {
        "shape": np.asarray((m, n), dtype=np.int64),
        "indptr": np.ctypeslib.as_array(output.offsets, shape=(m + 1,)).copy(),
        "indices": np.ctypeslib.as_array(output.columns, shape=(output.nnz,)).copy(),
        "data": np.ctypeslib.as_array(output.values, shape=(output.nnz,)).copy(),
      }
    finally:
      free_output(ct.byref(output))
    return {**_csr_arrays(A, "_a"), **_csr_arrays(B, "_b"), **arrays}

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


def run_mttkrp(A_path, B_path, C_path, repeat, device, warmup, *, reference=False):
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
  samples = None if reference else np.empty(repeat, dtype=np.float32)
  output = np.empty((m, rank), dtype=np.float32) if reference else None

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
    samples.ctypes.data if samples is not None else None,
    output.ctypes.data if output is not None else None,
  )
  if error:
    raise RuntimeError(error.decode())

  if reference:
    return {
      "shape": np.asarray(shape, dtype=np.int64),
      "positions_j": positions_j, "coordinates_j": coordinates_j,
      "positions_k": positions_k, "coordinates_k": coordinates_k,
      "values": values, "b": b, "c": c, "d": output,
    }

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
