#!/usr/bin/env python3
"""Benchmark Nacho SpGEMM with CUDA events and record every execution in CSV."""

import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.args import parse_run_args
from benchmarks.result import BenchmarkResult, InputInfo


OPERATIONS = ("spgemm",)


class HostCsr(ct.Structure):
    _fields_ = [
        ("rows", ct.c_int),
        ("cols", ct.c_int),
        ("nnz", ct.c_int64),
        ("offsets", ct.c_void_p),
        ("columns", ct.c_void_p),
        ("values", ct.c_void_p),
    ]

    @classmethod
    def from_matrix(cls, matrix):
        return cls(
            *matrix.shape,
            matrix.nnz,
            matrix.indptr.ctypes.data,
            matrix.indices.ctypes.data,
            matrix.data.ctypes.data,
        )


def load_matrix(path, np, io, sparse):
    matrix = io.mmread(path, spmatrix=False)
    if np.iscomplexobj(matrix):
        raise ValueError(f"Complex matrices are unsupported: {path}")
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if min(matrix.shape) < 1 or max(*matrix.shape, matrix.nnz) > 2**31 - 1:
        raise ValueError(
            "Matrix dimensions must be positive; dimensions and nnz must fit int32"
        )
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"Matrix values must be finite float32 numbers: {path}")
    matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    matrix.indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=np.float32)
    return matrix


def main():
    args = parse_run_args(OPERATIONS)
    blocks = 256
    threads_per_block = 256
    output_path = Path("generated/nacho") / "spgemm.csv"

    from benchmarks.spgemm import config

    for A_filename, B_filename in config.CASES:
        A_path = Path("dataset") / A_filename
        B_path = Path("dataset") / B_filename
        run_spgemm(
            A_path,
            B_path,
            args.repeat,
            args.device,
            args.warmup,
            blocks,
            threads_per_block,
            output_path,
        )


def run_spgemm(
    A_path, B_path, repeat, device, warmup, blocks, threads_per_block, output_path
):
    library_path = Path("generated/nacho") / "libnacho_spgemm.so"
    if not library_path.is_file():
        raise RuntimeError(
            f"Missing benchmark library: {library_path}\n"
            "Build it first: python -m systems.nacho.build"
        )
    try:
        import numpy as np
        from scipy import io, sparse
    except ImportError as error:
        raise RuntimeError(
            f"Missing dependency: {error}. Install numpy and scipy."
        ) from error

    a = load_matrix(A_path, np, io, sparse)
    b = load_matrix(B_path, np, io, sparse)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible SpGEMM shapes: {a.shape} and {b.shape}")
    host_a, host_b = HostCsr.from_matrix(a), HostCsr.from_matrix(b)
    samples = np.empty(repeat, dtype=np.float32)
    library = ct.CDLL(str(library_path))
    try:
        benchmark = library.spruce_benchmark_spgemm
    except AttributeError as error:
        raise RuntimeError(
            "Rebuild the library: python -m systems.nacho.build"
        ) from error
    benchmark.restype = ct.c_char_p
    benchmark.argtypes = [
        ct.c_int,
        ct.POINTER(HostCsr),
        ct.POINTER(HostCsr),
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_void_p,
    ]
    run_utc = datetime.now(timezone.utc).isoformat()
    error = benchmark(
        device,
        ct.byref(host_a),
        ct.byref(host_b),
        blocks,
        threads_per_block,
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
        inputs={
            "A": InputInfo(str(A_path.resolve()), a.shape, a.nnz),
            "B": InputInfo(str(B_path.resolve()), b.shape, b.nnz),
        },
        device=device,
        warmup=warmup,
        samples_ms=samples.tolist(),
        scope="generated_spgemm_call",
    )
    result.print_summary()
    result.write_csv(output_path)


if __name__ == "__main__":
    main()
