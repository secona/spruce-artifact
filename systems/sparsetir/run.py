"""Benchmark SparseTIR's prebuilt GPU kernels on configured inputs."""

import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.args import parse_run_args
from benchmarks.inputs import load_csf, load_csr, load_dense
from benchmarks.result import BenchmarkResult


OPERATIONS = ("spmv", "sddmm", "mttkrp")


def run_spmv(A_path, x_path, repeat, device, warmup):
  library_path = Path("generated") / "sparsetir" / "libsparsetir_spmv.so"
  assert library_path.is_file()

  a = load_csr(A_path)
  x = load_dense(x_path, a.shape[1], 1)
  output = np.zeros(a.shape[0], dtype=np.float32)
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
    ct.c_void_p,
  ]
  run_utc = datetime.now(timezone.utc).isoformat()
  error = benchmark(
    device,
    *a.shape,
    a.nnz,
    a.indptr.ctypes.data,
    a.indices.ctypes.data,
    a.data.ctypes.data,
    x.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
    output.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="sparsetir",
    operation="spmv",
    operand_names=[A_path.name, x_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
    scope="execution",
  )
  result.print_summary()
  result.write_csv()


def run_sddmm(A_path, B_path, C_path, repeat, device, warmup):
  library_path = Path("generated") / "sparsetir" / "libsparsetir_sddmm.so"
  assert library_path.is_file()

  a = load_csr(A_path)
  b = load_dense(B_path, a.shape[0], 2)
  c = load_dense(C_path, a.shape[1], 2)
  if b.shape[1] != c.shape[1]:
    raise ValueError("SDDMM factor ranks must match")
  output = np.zeros(a.nnz, dtype=np.float32)
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
    ct.c_void_p,
  ]
  run_utc = datetime.now(timezone.utc).isoformat()
  error = benchmark(
    device,
    *a.shape,
    b.shape[1],
    a.nnz,
    a.indptr.ctypes.data,
    a.indices.ctypes.data,
    a.data.ctypes.data,
    b.ctypes.data,
    c.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
    output.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="sparsetir",
    operation="sddmm",
    operand_names=[A_path.name, B_path.name, C_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
    scope="execution",
  )
  result.print_summary()
  result.write_csv()


def run_mttkrp(A_path, B_path, C_path, repeat, device, warmup):
  library_path = Path("generated") / "sparsetir" / "libsparsetir_mttkrp.so"
  assert library_path.is_file()

  shape, positions_j, coordinates_j, positions_k, coordinates_k, values = load_csf(A_path)
  # The shared CSF loader uses uint64 for MLIR; SparseTIR's existing kernels
  # declare int32 sparse indices. The loader bounds all coordinates and counts.
  positions_j = np.ascontiguousarray(positions_j, dtype=np.int32)
  coordinates_j = np.ascontiguousarray(coordinates_j, dtype=np.int32)
  positions_k = np.ascontiguousarray(positions_k, dtype=np.int32)
  coordinates_k = np.ascontiguousarray(coordinates_k, dtype=np.int32)
  b = load_dense(B_path, shape[1], 2)
  c = load_dense(C_path, shape[2], 2)
  if b.shape[1] != c.shape[1]:
    raise ValueError("MTTKRP factor ranks must match")
  output = np.zeros((shape[0], b.shape[1]), dtype=np.float32)
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
    ct.c_int64,
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
  run_utc = datetime.now(timezone.utc).isoformat()
  error = benchmark(
    device,
    *shape,
    b.shape[1],
    coordinates_j.size,
    values.size,
    positions_j.ctypes.data,
    coordinates_j.ctypes.data,
    positions_k.ctypes.data,
    coordinates_k.ctypes.data,
    values.ctypes.data,
    b.ctypes.data,
    c.ctypes.data,
    warmup,
    repeat,
    samples.ctypes.data,
    output.ctypes.data,
  )
  if error:
    raise RuntimeError(error.decode())

  result = BenchmarkResult(
    run_utc=run_utc,
    backend="sparsetir",
    operation="mttkrp",
    operand_names=[A_path.name, B_path.name, C_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
    scope="execution",
  )
  result.print_summary()
  result.write_csv()


def main():
  args = parse_run_args(OPERATIONS)

  for operation in (args.operation,) if args.operation else OPERATIONS:
    if operation == "spmv":
      from benchmarks.spmv import config
      run = run_spmv
    elif operation == "sddmm":
      from benchmarks.sddmm import config
      run = run_sddmm
    else:
      from benchmarks.mttkrp import config
      run = run_mttkrp

    for filenames in config.CASES:
      paths = [Path("dataset") / filename for filename in filenames]
      run(*paths, args.repeat, args.device, args.warmup)


if __name__ == "__main__":
  main()
