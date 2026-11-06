import argparse
import os
from pathlib import Path
import shlex
import subprocess


OPERATIONS = ("spmv", "sddmm", "spgemm")


def build(operation):
    assert operation in OPERATIONS

    backend_dir = Path("systems/cusparse")
    generated_dir = Path("generated") / "cusparse"
    generated_dir.mkdir(parents=True, exist_ok=True)
    library = generated_dir / f"libcusparse_{operation}.so"

    subprocess.run(
        [
            *shlex.split(os.environ.get("NVCC", "nvcc")),
            "-O3",
            "-std=c++17",
            "-shared",
            "-Xcompiler=-fPIC",
            "-I",
            str(backend_dir / "kernels"),
            str(backend_dir / ("benchmark_spgemm.cu" if operation == "spgemm" else "benchmark.cu")),
            str(backend_dir / "kernels" / f"{operation}.cu"),
            "-lcusparse",
            "-o",
            str(library),
        ],
        check=True
    )

    return library


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=OPERATIONS)
    args = parser.parse_args()

    for operation in (args.operation,) if args.operation else OPERATIONS:
        print(build(operation))
