import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import io, sparse

INT_MAX = 2**31 - 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("spmv", "sddmm", "spgemm", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    assert 0 <= args.warmup <= INT_MAX
    assert 1 <= args.repeat <= INT_MAX
    assert 0 <= args.device <= INT_MAX

    if args.operation == "spmv" or args.operation == "all":
        from benchmarks.spmv import config
        for (A_filename, x_filename) in config.CASES:
            A_path = Path("dataset") / A_filename
            x_path = Path("dataset") / x_filename
            run_spmv(A_path, x_path, args.repeat, args.device, args.warmup)

    if args.operation == "spgemm" or args.operation == "all":
        from benchmarks.spgemm import config
        for (A_filename, B_filename) in config.CASES:
            A_path = Path("dataset") / A_filename
            B_path = Path("dataset") / B_filename
            run_spgemm(A_path, B_path, args.repeat, args.device, args.warmup)


def run_spmv(A_path, x_path, repeat, device, warmup):
    run_utc = datetime.now(timezone.utc).isoformat()

    library_path = Path("generated") / "cusparse" / "libcusparse_spmv.so"
    assert library_path.is_file()

    matrix = io.mmread(A_path, spmatrix=False)
    assert not np.iscomplexobj(matrix)

    A = sparse.csr_matrix(matrix, dtype=np.float32)
    A.sum_duplicates()
    A.sort_indices()
    m, n = A.shape

    assert min(m, n) >= 1
    assert max(m, n, A.nnz) <= INT_MAX
    assert np.all(np.isfinite(A.data))

    offsets = np.ascontiguousarray(A.indptr, dtype=np.int32)
    columns = np.ascontiguousarray(A.indices, dtype=np.int32)
    values = np.ascontiguousarray(A.data, dtype=np.float32)
    x = np.load(x_path)
    samples = np.empty(repeat, dtype=np.float32)

    library = ct.CDLL(str(library_path))
    benchmark = library.spruce_benchmark_spmv
    benchmark.restype = ct.c_char_p
    benchmark.argtypes = [
        ct.c_int, ct.c_int, ct.c_int, ct.c_int64,
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_void_p,
    ]

    error = benchmark(
        device, m, n, A.nnz, offsets.ctypes.data, columns.ctypes.data,
        values.ctypes.data, x.ctypes.data, warmup, repeat, samples.ctypes.data,
    )
    if error:
        raise RuntimeError(error.decode())

    result = {
        "operation": "spmv",
        "matrix": str(A_path.resolve()),
        "shape": [m, n],
        "nnz": A.nnz,
        "device": device,
        "warmup": warmup,
        "repeat": repeat,
        "timing": "cuda_events",
        "scope": "execution",
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "samples_ms": samples.tolist(),
    }

    print(f"SpMV: shape={A.shape}, nnz={A.nnz}")
    print(f"CUDA events: median={result['median_ms']:.6f} ms, "
          f"mean={result['mean_ms']:.6f} ms, min={result['min_ms']:.6f} ms "
          f"({repeat} samples)")

    output = Path("generated/cusparse/spmv.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "run_utc", "operation", "matrix", "rows", "cols", "nnz",
            "device", "warmup", "iteration", "elapsed_ms",
        ])
        if out.tell() == 0:
            writer.writeheader()
        for iteration, elapsed in enumerate(samples, start=1):
            writer.writerow({
                "run_utc": run_utc,
                "operation": "spmv",
                "matrix": result["matrix"],
                "rows": m,
                "cols": n,
                "nnz": A.nnz,
                "device": device,
                "warmup": warmup,
                "iteration": iteration,
                "elapsed_ms": float(elapsed),
            })
    print(f"Recorded {repeat} executions: {output}")


def run_spgemm(A_path, B_path, repeat, device, warmup):
    run_utc = datetime.now(timezone.utc).isoformat()

    library_path = Path("generated") / "cusparse" / "libcusparse_spgemm.so"
    assert library_path.is_file()

    matrices = []
    for path in (A_path, B_path):
        matrix = io.mmread(path, spmatrix=False)
        assert not np.iscomplexobj(matrix)

        matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        matrix.sum_duplicates()
        matrix.sort_indices()

        assert min(matrix.shape) >= 1
        assert max(*matrix.shape, matrix.nnz) <= INT_MAX
        assert np.all(np.isfinite(matrix.data))

        matrices.append(matrix)

    A, B = matrices
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
        ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ct.c_int64, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int64, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_void_p,
    ]

    error = benchmark(
        device, m, k, n,
        A.nnz, offsets_a.ctypes.data, columns_a.ctypes.data, values_a.ctypes.data,
        B.nnz, offsets_b.ctypes.data, columns_b.ctypes.data, values_b.ctypes.data,
        warmup, repeat, samples.ctypes.data,
    )
    if error:
        raise RuntimeError(error.decode())

    result = {
        "operation": "spgemm",
        "matrix_a": str(A_path.resolve()),
        "matrix_b": str(B_path.resolve()),
        "shape_a": [m, k],
        "shape_b": [k, n],
        "nnz_a": A.nnz,
        "nnz_b": B.nnz,
        "device": device,
        "warmup": warmup,
        "repeat": repeat,
        "timing": "cuda_events",
        "scope": "execution",
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "samples_ms": samples.tolist(),
    }

    print(f"SpGEMM: A={A.shape}, nnzA={A.nnz}, B={B.shape}, nnzB={B.nnz}")
    print(f"CUDA events: median={result['median_ms']:.6f} ms, "
          f"mean={result['mean_ms']:.6f} ms, min={result['min_ms']:.6f} ms "
          f"({repeat} samples)")

    output = Path("generated/cusparse/spgemm.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "run_utc", "operation", "matrix_a", "matrix_b", "m", "k", "n",
            "nnz_a", "nnz_b", "device", "warmup", "iteration", "elapsed_ms",
        ])
        if out.tell() == 0:
            writer.writeheader()
        for iteration, elapsed in enumerate(samples, start=1):
            writer.writerow({
                "run_utc": run_utc,
                "operation": "spgemm",
                "matrix_a": result["matrix_a"],
                "matrix_b": result["matrix_b"],
                "m": m,
                "k": k,
                "n": n,
                "nnz_a": A.nnz,
                "nnz_b": B.nnz,
                "device": device,
                "warmup": warmup,
                "iteration": iteration,
                "elapsed_ms": float(elapsed),
            })
    print(f"Recorded {repeat} executions: {output}")


if __name__ == "__main__":
    main()
