#!/usr/bin/env python3
"""Benchmark Nacho SpGEMM with CUDA events and record every execution in CSV."""

import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import sys


class HostCsr(ct.Structure):
    _fields_ = [
        ("rows", ct.c_int), ("cols", ct.c_int), ("nnz", ct.c_int64),
        ("offsets", ct.c_void_p), ("columns", ct.c_void_p), ("values", ct.c_void_p),
    ]

    @classmethod
    def from_matrix(cls, matrix):
        return cls(*matrix.shape, matrix.nnz, matrix.indptr.ctypes.data,
                   matrix.indices.ctypes.data, matrix.data.ctypes.data)


def load_matrix(path, np, io, sparse):
    matrix = io.mmread(path)
    if np.iscomplexobj(matrix):
        raise ValueError(f"Complex matrices are unsupported: {path}")
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if min(matrix.shape) < 1 or max(*matrix.shape, matrix.nnz) > 2**31 - 1:
        raise ValueError("Matrix dimensions must be positive; dimensions and nnz must fit int32")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"Matrix values must be finite float32 numbers: {path}")
    matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    matrix.indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=np.float32)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("spgemm",))
    parser.add_argument("--matrix", type=Path, required=True, help="Matrix Market input A")
    parser.add_argument("--matrix-b", type=Path, help="Matrix Market input B (default: A, for square A)")
    # Defaults from third_party/nacho/benchmarks/config.py.
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=14)
    parser.add_argument("--trim", type=int, default=2, help="Samples dropped from each tail for trimmed mean only")
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--threads-per-block", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("generated/nacho") / "spgemm.csv",
                        help="Append one CSV row per timed execution (default: generated/nacho/spgemm.csv)")
    args = parser.parse_args()
    int_max = 2**31 - 1
    if not 0 <= args.warmup <= int_max or not 1 <= args.repeat <= int_max or not 0 <= args.device <= int_max:
        parser.error("warmup/device must be nonnegative and repeat positive; all must fit int32")
    if args.trim < 0 or 2 * args.trim >= args.repeat:
        parser.error("trim must be nonnegative and repeat must exceed 2 * trim (use --trim 0 for short runs)")
    if args.blocks < 1 or args.threads_per_block < 1 or args.blocks * args.threads_per_block >= int_max:
        parser.error("blocks and threads-per-block must be positive; their product must be less than int32 max")
    library_path = Path("generated/nacho") / "libnacho_spgemm.so"
    if not library_path.is_file():
        parser.exit(1, f"Missing benchmark library: {library_path}\n"
                    "Build it first: python -m systems.nacho.build\n")
    try:
        import numpy as np
        from scipy import io, sparse
    except ImportError as error:
        parser.exit(1, f"Missing dependency: {error}. Install numpy and scipy.\n")

    a = load_matrix(args.matrix, np, io, sparse)
    b = load_matrix(args.matrix_b, np, io, sparse) if args.matrix_b else a
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible SpGEMM shapes: {a.shape} and {b.shape}; provide a compatible --matrix-b")
    host_a, host_b = HostCsr.from_matrix(a), HostCsr.from_matrix(b)
    samples = np.empty(args.repeat, dtype=np.float32)
    library = ct.CDLL(str(library_path))
    try:
        benchmark = library.spruce_benchmark_spgemm
    except AttributeError as error:
        raise RuntimeError("Rebuild the library: python -m systems.nacho.build") from error
    benchmark.restype = ct.c_char_p
    benchmark.argtypes = [
        ct.c_int, ct.POINTER(HostCsr), ct.POINTER(HostCsr),
        ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p,
    ]
    run_utc = datetime.now(timezone.utc).isoformat()
    error = benchmark(args.device, ct.byref(host_a), ct.byref(host_b),
                      args.blocks, args.threads_per_block, args.warmup, args.repeat,
                      samples.ctypes.data)
    if error:
        raise RuntimeError(error.decode())
    trimmed = np.sort(samples)[args.trim:args.repeat - args.trim]
    print(f"Nacho SpGEMM: A={a.shape}, nnzA={a.nnz}, B={b.shape}, nnzB={b.nnz}")
    print(f"CUDA events: median={float(np.median(samples)):.6f} ms, "
          f"mean={float(np.mean(samples)):.6f} ms, min={float(np.min(samples)):.6f} ms, "
          f"trimmed mean={float(np.mean(trimmed)):.6f} ms ({args.repeat} samples)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "run_utc", "backend", "operation", "matrix_a", "matrix_b",
            "m", "k", "n", "nnz_a", "nnz_b", "device", "blocks", "threads_per_block",
            "warmup", "trim", "scope", "iteration", "elapsed_ms",
        ])
        if output.tell() == 0:
            writer.writeheader()
        for iteration, elapsed in enumerate(samples, start=1):
            writer.writerow({
                "run_utc": run_utc, "backend": "nacho", "operation": "spgemm",
                "matrix_a": str(args.matrix.resolve()),
                "matrix_b": str((args.matrix_b or args.matrix).resolve()),
                "m": a.shape[0], "k": a.shape[1], "n": b.shape[1],
                "nnz_a": a.nnz, "nnz_b": b.nnz, "device": args.device,
                "blocks": args.blocks, "threads_per_block": args.threads_per_block,
                "warmup": args.warmup, "trim": args.trim, "scope": "generated_spgemm_call",
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
