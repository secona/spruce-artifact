import argparse
import os
import subprocess
from pathlib import Path


def build(operation, llvm_build_dir):
    backend_dir = Path("systems/mlir_sparse_tensor")
    kernels_dir = backend_dir / "kernels"
    generated_dir = Path("generated") / "mlir_sparse_tensor"
    generated_dir.mkdir(parents=True, exist_ok=True)

    mlir_opt = llvm_build_dir / "bin" / "mlir-opt"
    mlir_translate = llvm_build_dir / "bin" / "mlir-translate"
    llc = llvm_build_dir / "bin" / "llc"

    commands = [
        [
            str(mlir_opt),
            str(kernels_dir / f"{operation}.mlir"),
            # f"--pass-pipeline={GPU_PIPELINE}",
            "--sparsifier=gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_75 enable-gpu-libgen=false parallelization-strategy=dense-outer-loop",
            "-o",
            str(generated_dir / f"{operation}.mlir"),
        ],
        [
            str(mlir_translate),
            str(generated_dir / f"{operation}.mlir"),
            "--mlir-to-llvmir",
            "-o",
            str(generated_dir / f"{operation}.ll"),
        ],
        [
            str(llc),
            "-O3",
            "-filetype=obj",
            "-relocation-model=pic",
            str(generated_dir / f"{operation}.ll"),
            "-o",
            str(generated_dir / f"{operation}.o")
        ],
    ]

    for command in commands:
        print("[Running]", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    return generated_dir / f"{operation}.mlir"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("spmv", "mttkrp", "sddmm", "spgemm"))
    args = parser.parse_args()

    llvm_build_dir = os.getenv("LLVM_DIR")
    assert llvm_build_dir

    for operation in (args.operation,) if args.operation else ("spmv", "mttkrp"):
        print("[Built]", build(operation, Path(llvm_build_dir)))
