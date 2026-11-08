"""Benchmark MLIR SparseTensor's GPU kernels on configured inputs."""

import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.args import parse_run_args
from benchmarks.inputs import check_output, load_csf, load_csr, load_dense
from benchmarks.result import BenchmarkResult


OPERATIONS = ("spmv", "mttkrp")


def load_library(operation, device):
  library_path = Path("generated/mlir_sparse_tensor") / f"libmlir_sparse_tensor_{operation}.so"
  assert library_path.is_file()
  # Validate before loading the module: its constructor initializes CUDA.
  driver = ct.CDLL("libcuda.so.1")
  driver.cuInit.argtypes = [ct.c_uint]
  driver.cuDeviceGetCount.argtypes = [ct.POINTER(ct.c_int)]
  count = ct.c_int()
  assert driver.cuInit(0) == 0
  assert driver.cuDeviceGetCount(ct.byref(count)) == 0
  assert 0 <= device < count.value
  runtime_path = library_path.parent / "cuda-runtime.txt"
  assert runtime_path.is_file()
  runtime = ct.CDLL(runtime_path.read_text().strip())
  runtime.mgpuSetDefaultDevice.argtypes = [ct.c_int32]
  runtime.mgpuSetDefaultDevice.restype = None
  runtime.mgpuSetDefaultDevice(device)
  return ct.CDLL(str(library_path))


def run_spmv(A_path, x_path, repeat, device, warmup):
  library = load_library("spmv", device)
  a = load_csr(A_path, np.uint64)
  x = load_dense(x_path, a.shape[1], 1)
  output = np.zeros(a.shape[0], dtype=np.float32)
  samples = np.empty(repeat, dtype=np.float64)
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
  assert not error
  reference = a.astype(np.float64) @ x.astype(np.float64)
  absolute_products = abs(a.astype(np.float64)) @ abs(x.astype(np.float64))
  check_output(output, reference, absolute_products)
  result = BenchmarkResult(
    run_utc=run_utc,
    backend="mlir_sparse_tensor",
    operation="spmv",
    operand_names=[A_path.name, x_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
    timing="cuda_events",
    scope="generated_call_including_transfers",
  )
  result.print_summary()
  result.write_csv()


def run_mttkrp(A_path, B_path, C_path, repeat, device, warmup):
  library = load_library("mttkrp", device)
  shape, positions_j, coordinates_j, positions_k, coordinates_k, values = load_csf(A_path)
  b = load_dense(B_path, shape[1], 2)
  c = load_dense(C_path, shape[2], 2)
  assert b.shape[1] == c.shape[1]
  output = np.zeros((shape[0], b.shape[1]), dtype=np.float32)
  samples = np.empty(repeat, dtype=np.float64)
  benchmark = library.spruce_benchmark_mttkrp
  benchmark.restype = ct.c_char_p
  benchmark.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
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
  run_utc = datetime.now(timezone.utc).isoformat()
  error = benchmark(
    device,
    *shape,
    b.shape[1],
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
  assert not error
  result = BenchmarkResult(
    run_utc=run_utc,
    backend="mlir_sparse_tensor",
    operation="mttkrp",
    operand_names=[A_path.name, B_path.name, C_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
    timing="cuda_events",
    scope="generated_call_including_transfers",
  )
  result.print_summary()
  result.write_csv()


def main():
  args = parse_run_args(OPERATIONS)

  for operation in (args.operation,) if args.operation else OPERATIONS:
    if operation == "spmv":
      from benchmarks.spmv import config
      run = run_spmv
    else:
      from benchmarks.mttkrp import config
      run = run_mttkrp

    for filenames in config.CASES:
      paths = [Path("dataset") / filename for filename in filenames]
      run(*paths, args.repeat, args.device, args.warmup)


if __name__ == "__main__":
  main()
