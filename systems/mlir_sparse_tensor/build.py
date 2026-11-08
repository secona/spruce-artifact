import os
import shlex
import subprocess
from pathlib import Path
from benchmarks.args import parse_build_args


OPERATIONS = ("spmv", "mttkrp")


def main():
  args = parse_build_args(OPERATIONS)

  llvm_build_dir = os.getenv("LLVM_DIR")
  assert llvm_build_dir

  for operation in (args.operation,) if args.operation else OPERATIONS:
    build(operation, Path(llvm_build_dir), args.gpu_arch)


def build(operation, llvm_build_dir, gpu_arch):
  assert operation in OPERATIONS

  cuda_root = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))

  backend_dir = Path("systems/mlir_sparse_tensor")
  kernels_dir = backend_dir / "kernels"
  generated_dir = Path("generated") / "mlir_sparse_tensor"
  generated_dir.mkdir(parents=True, exist_ok=True)

  mlir_opt = llvm_build_dir / "bin" / "mlir-opt"
  mlir_translate = llvm_build_dir / "bin" / "mlir-translate"
  llc = llvm_build_dir / "bin" / "llc"

  include_dirs = [llvm_build_dir / "include"]
  if mlir_include_dir := os.environ.get("MLIR_INCLUDE_DIR"):
    include_dirs.append(Path(mlir_include_dir))

  runtime_dir = (llvm_build_dir / "lib").resolve()
  library = generated_dir / f"libmlir_sparse_tensor_{operation}.so"

  commands = [
    (
      [
        str(mlir_opt),
        str(kernels_dir / f"{operation}.mlir"),
        f"--sparsifier=gpu-triple=nvptx64-nvidia-cuda gpu-chip={gpu_arch} enable-gpu-libgen=false parallelization-strategy=dense-outer-loop",
        "-o",
        str(generated_dir / f"{operation}.mlir"),
      ],
      str(generated_dir / f"{operation}.mlir"),
    ),
    (
      [
        str(mlir_translate),
        str(generated_dir / f"{operation}.mlir"),
        "--mlir-to-llvmir",
        "-o",
        str(generated_dir / f"{operation}.ll"),
      ],
      str(generated_dir / f"{operation}.ll"),
    ),
    (
      [
        str(llc),
        "-O3",
        "-filetype=obj",
        "-relocation-model=pic",
        str(generated_dir / f"{operation}.ll"),
        "-o",
        str(generated_dir / f"{operation}.o"),
      ],
      str(generated_dir / f"{operation}.o"),
    ),
    (
      [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *[flag for path in include_dirs for flag in ("-I", str(path))],
        "-I",
        str(cuda_root / "include"),
        str(backend_dir / f"benchmark_{operation}.cpp"),
        str(generated_dir / f"{operation}.o"),
        "-L",
        str(runtime_dir),
        f"-Wl,-rpath,{runtime_dir}",
        "-lmlir_c_runner_utils",
        "-lmlir_cuda_runtime",
        "-L",
        str(cuda_root / "lib"),
        "-L",
        str(cuda_root / "lib64"),
        "-lcudart",
        "-o",
        str(library),
      ],
      str(library),
    ),
  ]

  for command, out in commands:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print("built:", out)

  # The runner selects the runtime device before the generated module's
  # constructor loads its GPU binary into that device's CUDA context.
  (generated_dir / "cuda-runtime.txt").write_text(
    str(runtime_dir / "libmlir_cuda_runtime.so")
  )


if __name__ == "__main__":
  main()
