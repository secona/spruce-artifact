"""Benchmark Nacho's generated SpGEMM kernel on configured inputs."""

import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.args import parse_run_args
from benchmarks.inputs import load_csr
from benchmarks.result import BenchmarkResult


OPERATIONS = ("spgemm",)


def run_spgemm(A_path, B_path, repeat, device, warmup):
  run_utc = datetime.now(timezone.utc).isoformat()

  library_path = Path("generated") / "nacho" / "libnacho_spgemm.so"
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
    backend="nacho",
    operation="spgemm",
    operand_names=[A_path.name, B_path.name],
    device=device,
    warmup=warmup,
    samples_ms=samples.tolist(),
    scope="generated_spgemm_call",
  )
  result.print_summary()
  result.write_csv()


def main():
  args = parse_run_args(OPERATIONS)

  for operation in (args.operation,) if args.operation else OPERATIONS:
    from benchmarks.spgemm import config

    for filenames in config.CASES:
      paths = [Path("dataset") / filename for filename in filenames]
      run_spgemm(*paths, args.repeat, args.device, args.warmup)


if __name__ == "__main__":
  main()
