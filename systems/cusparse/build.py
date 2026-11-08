import os
from pathlib import Path
import shlex
import subprocess
from benchmarks.args import parse_build_args


OPERATIONS = ("spmv", "sddmm", "spgemm")


def main():
    args = parse_build_args(OPERATIONS)

    for operation in (args.operation,) if args.operation else OPERATIONS:
        print(build(operation, args.gpu_arch))


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
            str(backend_dir / f"benchmark_{operation}.cu"),
            str(backend_dir / "kernels" / f"{operation}.cu"),
            "-lcusparse",
            "-o",
            str(library),
        ],
        check=True
    )

    return library


if __name__ == "__main__":
    main()
