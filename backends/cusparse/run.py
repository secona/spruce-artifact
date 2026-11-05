#!/usr/bin/env python3
"""Run and benchmark cuSPARSE SpMV using CUDA events."""

import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("spmv", "sddmm", "spgemm"))
    parser.add_argument("--matrix", type=Path, required=True, help="Matrix Market input A")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "generated" / "cusparse" / "spmv.csv",
                        help="Append one CSV row per timed repetition (default: generated/cusparse/spmv.csv)")
    args = parser.parse_args()
    run_utc = datetime.now(timezone.utc).isoformat()
    int_max = 2**31 - 1
    if not 0 <= args.warmup <= int_max or not 1 <= args.repeat <= int_max or not 0 <= args.device <= int_max:
        parser.error("warmup/device must be nonnegative and repeat positive; all must fit int32")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    library_path = Path(__file__).resolve().parents[2] / "generated" / "cusparse" / "libcusparse_spmv.so"
    if not library_path.is_file():
        parser.exit(1, f"Missing benchmark library: {library_path}\n"
                    "Build it first: python backends/cusparse/build.py\n")
    try:
        import numpy as np
        from scipy import io, sparse
    except ImportError as error:
        parser.exit(1, f"Missing dependency: {error}. Install numpy and scipy.\n")

    matrix = io.mmread(args.matrix)
    if np.iscomplexobj(matrix):
        raise ValueError("Complex matrices are unsupported")
    a = sparse.csr_matrix(matrix, dtype=np.float32)
    a.sum_duplicates()
    a.sort_indices()
    m, n = a.shape
    if min(m, n) < 1 or max(m, n, a.nnz) > int_max:
        raise ValueError("Matrix dimensions must be positive; dimensions and nnz must fit int32")
    if not np.all(np.isfinite(a.data)):
        raise ValueError("Matrix values must be finite float32 numbers")
    offsets = np.ascontiguousarray(a.indptr, dtype=np.int32)
    columns = np.ascontiguousarray(a.indices, dtype=np.int32)
    values = np.ascontiguousarray(a.data, dtype=np.float32)
    x = np.random.default_rng(args.seed).random(n, dtype=np.float32)
    samples = np.empty(args.repeat, dtype=np.float32)

    library = ct.CDLL(str(library_path))
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
        args.device, m, n, a.nnz, offsets.ctypes.data, columns.ctypes.data,
        values.ctypes.data, x.ctypes.data, args.warmup, args.repeat, samples.ctypes.data,
    )
    if error:
        raise RuntimeError(error.decode())
    result = {
        "operation": "spmv", "matrix": str(args.matrix.resolve()),
        "shape": [m, n], "nnz": a.nnz, "device": args.device,
        "seed": args.seed, "warmup": args.warmup, "repeat": args.repeat,
        "timing": "cuda_events", "scope": "execution",
        "median_ms": float(np.median(samples)), "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)), "max_ms": float(np.max(samples)),
        "samples_ms": samples.tolist(),
    }
    print(f"SpMV: shape={a.shape}, nnz={a.nnz}")
    print(f"CUDA events: median={result['median_ms']:.6f} ms, "
          f"mean={result['mean_ms']:.6f} ms, min={result['min_ms']:.6f} ms "
          f"({args.repeat} samples)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "run_utc", "operation", "matrix", "rows", "cols", "nnz",
            "device", "seed", "warmup", "iteration", "elapsed_ms",
        ])
        if output.tell() == 0:
            writer.writeheader()
        for iteration, elapsed in enumerate(samples, start=1):
            writer.writerow({
                "run_utc": run_utc, "operation": "spmv", "matrix": result["matrix"],
                "rows": m, "cols": n, "nnz": a.nnz, "device": args.device,
                "seed": args.seed, "warmup": args.warmup,
                "iteration": iteration, "elapsed_ms": float(elapsed),
            })
    print(f"Recorded {args.repeat} executions: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
