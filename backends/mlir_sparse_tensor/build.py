import argparse
import os
import subprocess
from pathlib import Path

GPU_PIPELINE = (
    "builtin.module("
    "pre-sparsification-rewrite,"
    "sparse-reinterpret-map,"
    "sparsification{parallelization-strategy=dense-any-loop},"
    "sparse-gpu-codegen)"
)


def build(operation, mlir_opt):
    backend_dir = Path(__file__).resolve().parent
    generated_dir = backend_dir.parents[1] / "generated" / "mlir_sparse_tensor"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output = generated_dir / f"{operation}.mlir"
    try:
        subprocess.run(
            [
                str(mlir_opt),
                str(backend_dir / "kernels" / f"{operation}.mlir"),
                f"--pass-pipeline={GPU_PIPELINE}",
                "-o",
                str(output),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        output.unlink(missing_ok=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("spmv", "mttkrp", "sddmm", "spgemm"))
    args = parser.parse_args()

    llvm_build_dir = os.getenv("LLVM_DIR")
    assert llvm_build_dir

    mlir_opt = Path(llvm_build_dir).expanduser().resolve() / "bin" / "mlir-opt"
    assert mlir_opt.is_file()

    for operation in (args.operation,) if args.operation else ("spmv", "mttkrp"):
        print(build(operation, mlir_opt))
