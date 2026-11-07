"""Benchmark MLIR SparseTensor's GPU SpMV pipeline on configured inputs."""

import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.args import parse_run_args
from benchmarks.inputs import check_output, load_csr, load_dense
from benchmarks.result import BenchmarkResult, InputInfo


OPERATIONS = ("spmv",)


def run_spmv(A_path, x_path, repeat, device, warmup):
    library_path = Path("generated/mlir_sparse_tensor/libmlir_sparse_tensor_spmv.so")
    if not library_path.is_file():
        raise RuntimeError("Build the library first: python -m systems.mlir_sparse_tensor.build --operation spmv")
    # Validate before loading the module: its constructor initializes CUDA.
    driver = ct.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ct.c_uint]
    driver.cuDeviceGetCount.argtypes = [ct.POINTER(ct.c_int)]
    count = ct.c_int()
    if driver.cuInit(0) or driver.cuDeviceGetCount(ct.byref(count)):
        raise RuntimeError("Could not initialize the CUDA driver")
    if not 0 <= device < count.value:
        raise ValueError(f"CUDA device {device} is unavailable")
    a = load_csr(A_path, np.uint64)
    x = load_dense(x_path, a.shape[1], 1)
    output = np.zeros(a.shape[0], dtype=np.float32)
    samples = np.empty(repeat, dtype=np.float64)
    runtime_path = library_path.parent / "cuda-runtime.txt"
    if not runtime_path.is_file():
        raise RuntimeError("Rebuild the library: python -m systems.mlir_sparse_tensor.build --operation spmv")
    runtime = ct.CDLL(runtime_path.read_text().strip())
    runtime.mgpuSetDefaultDevice.argtypes = [ct.c_int32]
    runtime.mgpuSetDefaultDevice.restype = None
    runtime.mgpuSetDefaultDevice(device)
    library = ct.CDLL(str(library_path))
    benchmark = library.spruce_benchmark_spmv
    benchmark.restype = ct.c_char_p
    benchmark.argtypes = [
        ct.c_int, ct.c_int, ct.c_int, ct.c_int64,
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p,
    ]
    run_utc = datetime.now(timezone.utc).isoformat()
    error = benchmark(device, *a.shape, a.nnz,
                      a.indptr.ctypes.data, a.indices.ctypes.data, a.data.ctypes.data,
                      x.ctypes.data, warmup, repeat, samples.ctypes.data, output.ctypes.data)
    if error:
        raise RuntimeError(error.decode())
    reference = a.astype(np.float64) @ x.astype(np.float64)
    absolute_products = abs(a.astype(np.float64)) @ abs(x.astype(np.float64))
    check_output(output, reference, absolute_products)
    result = BenchmarkResult(
        run_utc=run_utc, backend="mlir_sparse_tensor", operation="spmv",
        inputs={
            "A": InputInfo(str(A_path.resolve()), a.shape, a.nnz),
            "x": InputInfo(str(x_path.resolve()), x.shape),
        },
        device=device, warmup=warmup, samples_ms=samples.tolist(),
        timing="wall_clock", scope="generated_call_including_transfers",
    )
    result.print_summary()
    result.write_csv("generated/mlir_sparse_tensor/spmv.csv")


def main():
    args = parse_run_args(OPERATIONS)

    from benchmarks.spmv import config
    for A_filename, x_filename in config.CASES:
        run_spmv(Path("dataset") / A_filename, Path("dataset") / x_filename,
                 args.repeat, args.device, args.warmup)


if __name__ == "__main__":
    main()
