import argparse
from pathlib import Path
import shutil


OPERATIONS = ("spmv", "sddmm", "spgemm")


def build(operation):
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")

    backend_dir = Path(__file__).resolve().parent
    output_dir = backend_dir.parents[1] / "generated" / "cusparse"
    output_dir.mkdir(parents=True, exist_ok=True)
    # cuSPARSE provides precompiled kernels; emit the CUDA API wrappers.
    for header in ("common.cuh", "kernels.h"):
        shutil.copyfile(backend_dir / "kernels" / header, output_dir / header)
    output = output_dir / f"{operation}.cu"
    # For the sake of same interface as other backends.
    shutil.copyfile(backend_dir / "kernels" / output.name, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=OPERATIONS)
    args = parser.parse_args()

    for operation in (args.operation,) if args.operation else OPERATIONS:
        print(build(operation))
