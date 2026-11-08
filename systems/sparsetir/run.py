"""Run configured SparseTIR benchmarks, specializing kernels to each input."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.args import parse_run_args
from benchmarks.inputs import check_output, load_csr, load_dense
from benchmarks.result import BenchmarkResult, InputInfo
from .build import compile_kernel


OPERATIONS = ("spmv", "sddmm")


def run(operation, paths, repeat, device, warmup, gpu_arch):
  import tvm

  a = load_csr(paths[0])
  inputs = {"A": InputInfo(str(paths[0].resolve()), a.shape, a.nnz)}
  if operation == "spmv":
    x = load_dense(paths[1], a.shape[1], 1)
    inputs["x"] = InputInfo(str(paths[1].resolve()), x.shape)
    output = np.zeros(a.shape[0], dtype=np.float32)
    arrays = [a.data, x, output, a.indptr, a.indices]
    shape = (*a.shape, a.nnz)
    reference = a.astype(np.float64) @ x.astype(np.float64)
    absolute_products = abs(a.astype(np.float64)) @ abs(x.astype(np.float64))
  else:
    b = load_dense(paths[1], a.shape[0], 2)
    c = load_dense(paths[2], a.shape[1], 2)
    if b.shape[1] != c.shape[1]:
      raise ValueError("SDDMM factor ranks must match")
    inputs.update(
      {
        "B": InputInfo(str(paths[1].resolve()), b.shape),
        "C": InputInfo(str(paths[2].resolve()), c.shape),
      }
    )
    output = np.zeros(a.nnz, dtype=np.float32)
    arrays = [a.data, b, c, output, a.indptr, a.indices]
    shape = (*a.shape, b.shape[1], a.nnz)
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    products = b[rows].astype(np.float64) * c[a.indices].astype(np.float64)
    reference = a.data.astype(np.float64) * products.sum(axis=1)
    absolute_products = abs(a.data.astype(np.float64)) * abs(products).sum(axis=1)

  dev = tvm.cuda(device)
  if not dev.exist:
    raise RuntimeError(f"CUDA device {device} is unavailable")
  compiled = compile_kernel(operation, gpu_arch, shape)
  # Lowered sparse buffers expose flat storage, including the dense factors.
  args = [tvm.nd.array(array.reshape(-1), dev) for array in arrays]
  output_index = 2 if operation == "spmv" else 3
  # Check one execution before collecting timings (also initializes the module).
  compiled[operation](*args)
  check_output(args[output_index].numpy(), reference, absolute_products)
  # TVM's evaluator performs one additional untimed invocation.
  for _ in range(max(0, warmup - 1)):
    compiled[operation](*args)
  dev.sync()
  run_utc = datetime.now(timezone.utc).isoformat()
  measurements = compiled.time_evaluator(operation, dev, number=1, repeat=repeat)(*args)
  result = BenchmarkResult(
    run_utc=run_utc,
    backend="sparsetir",
    operation=operation,
    inputs=inputs,
    device=device,
    warmup=warmup,
    samples_ms=[seconds * 1000 for seconds in measurements.results],
    scope="execution",
  )
  result.print_summary()
  result.write_csv(Path("generated/sparsetir") / f"{operation}.csv")


def main():
  args = parse_run_args(OPERATIONS)
  assert args.warmup >= 1
  gpu_arch = "sm_86"

  for operation in (args.operation,) if args.operation else OPERATIONS:
    if operation == "spmv":
      from benchmarks.spmv import config
    else:
      from benchmarks.sddmm import config
    for filenames in config.CASES:
      paths = [Path("dataset") / name for name in filenames]
      run(operation, paths, args.repeat, args.device, args.warmup, gpu_arch)


if __name__ == "__main__":
  main()
