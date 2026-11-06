#!/usr/bin/env python3
"""Run and benchmark cuSPARSE SpMV using CUDA events."""

import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
from scipy import io, sparse

INT_MAX = 2**31 - 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("spmv", "sddmm", "spgemm"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert 0 <= args.warmup <= INT_MAX
    assert 1 <= args.repeat <= INT_MAX
    assert 0 <= args.device <= INT_MAX
    assert args.seed >= 0

    library_path = Path("generated") / "cusparse" / "libcusparse_spmv.so"
    assert library_path.is_file()

    if args.operation == "spmv":
        from benchmarks.spmv import config
        for name in config.MATRICES:
            matrixfile = Path("dataset") / name
            run(matrixfile, args.seed, args.repeat, library_path, args.device, args.warmup)


def run(matrixfile, seed, repeat, libpath, device, warmup):
    run_utc = datetime.now(timezone.utc).isoformat()

    matrix = io.mmread(matrixfile, spmatrix=False)
    assert not np.iscomplexobj(matrix)

    a = sparse.csr_matrix(matrix, dtype=np.float32)
    a.sum_duplicates()
    a.sort_indices()
    m, n = a.shape
    if min(m, n) < 1 or max(m, n, a.nnz) > INT_MAX:
        raise ValueError("Matrix dimensions must be positive; dimensions and nnz must fit int32")
    if not np.all(np.isfinite(a.data)):
        raise ValueError("Matrix values must be finite float32 numbers")
    offsets = np.ascontiguousarray(a.indptr, dtype=np.int32)
    columns = np.ascontiguousarray(a.indices, dtype=np.int32)
    values = np.ascontiguousarray(a.data, dtype=np.float32)
    x = np.random.default_rng(seed).random(n, dtype=np.float32)
    samples = np.empty(repeat, dtype=np.float32)

    library = ct.CDLL(str(libpath))
    try:
        benchmark = library.spruce_benchmark_spmv
    except AttributeError as error:
        raise RuntimeError("Rebuild the library: python backends/cusparse/build.py") from error
    benchmark.restype = ct.c_char_p
    benchmark.argtypes = [
        ct.c_int, ct.c_int, ct.c_int, ct.c_int64,
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_void_p,
    ]
    error = benchmark(
        device, m, n, a.nnz, offsets.ctypes.data, columns.ctypes.data,
        values.ctypes.data, x.ctypes.data, warmup, repeat, samples.ctypes.data,
    )
    if error:
        raise RuntimeError(error.decode())
    result = {
        "operation": "spmv", "matrix": str(matrixfile.resolve()),
        "shape": [m, n], "nnz": a.nnz, "device": device,
        "seed": seed, "warmup": warmup, "repeat": repeat,
        "timing": "cuda_events", "scope": "execution",
        "median_ms": float(np.median(samples)), "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)), "max_ms": float(np.max(samples)),
        "samples_ms": samples.tolist(),
    }
    print(f"SpMV: shape={a.shape}, nnz={a.nnz}")
    print(f"CUDA events: median={result['median_ms']:.6f} ms, "
          f"mean={result['mean_ms']:.6f} ms, min={result['min_ms']:.6f} ms "
          f"({repeat} samples)")

    output = Path("generated/cusparse/spmv.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "run_utc", "operation", "matrix", "rows", "cols", "nnz",
            "device", "seed", "warmup", "iteration", "elapsed_ms",
        ])
        if out.tell() == 0:
            writer.writeheader()
        for iteration, elapsed in enumerate(samples, start=1):
            writer.writerow({
                "run_utc": run_utc, "operation": "spmv", "matrix": result["matrix"],
                "rows": m, "cols": n, "nnz": a.nnz, "device": device,
                "seed": seed, "warmup": warmup,
                "iteration": iteration, "elapsed_ms": float(elapsed),
            })
    print(f"Recorded {repeat} executions: {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
