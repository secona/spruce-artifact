import argparse
import os
from pathlib import Path
import shlex
import subprocess


OPERATIONS = ("spmv", "sddmm", "spgemm")


def build(operation, gpu_arch):
    assert operation in OPERATIONS

    backend_dir = Path("systems/cusparse")
    generated_dir = Path("generated") / "cusparse"
    generated_dir.mkdir(parents=True, exist_ok=True)
    library = generated_dir / f"libcusparse_{operation}.so"

    subprocess.run(
        [
            *shlex.split(os.environ.get("NVCC", "nvcc")),
            f"-arch={gpu_arch}",
            "-O3",
            "-std=c++17",
            "-shared",
            "-Xcompiler=-fPIC",
            "-I",
            str(backend_dir / "kernels"),
            str(backend_dir / ("benchmark.cu" if operation == "spmv" else f"benchmark_{operation}.cu")),
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
    parser.add_argument("--gpu-arch", default="sm_86")
    args = parser.parse_args()

    for operation in (args.operation,) if args.operation else OPERATIONS:
        print(build(operation, args.gpu_arch))
