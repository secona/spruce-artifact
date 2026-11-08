import os
import shlex
import subprocess
from pathlib import Path
from benchmarks.args import parse_build_args


OPERATIONS = ("spgemm",)


def main():
  args = parse_build_args(OPERATIONS)

  for operation in (args.operation,) if args.operation else OPERATIONS:
    build(operation, args.gpu_arch)


def build(operation, gpu_arch):
  assert operation in OPERATIONS

  backend_dir = Path("systems/nacho")
  nacho_dir = Path("third_party/nacho")
  generated_dir = Path("generated") / "nacho"
  generated_dir.mkdir(parents=True, exist_ok=True)
  build_dir = nacho_dir / "build"
  compiler = generated_dir / f"compile_{operation}"
  library = generated_dir / f"libnacho_{operation}.so"

  commands = [
    (
      [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-O2",
        "-std=c++20",
        "-I",
        str(nacho_dir / "include"),
        str(backend_dir / "kernels" / f"{operation}.cpp"),
        str(build_dir / "libnacho_library.a"),
        "-o",
        str(compiler),
      ],
      str(compiler),
    ),
    (
      [str(compiler), "--output-dir", str(generated_dir), "--kernels", operation],
      str(generated_dir / f"{operation}_gpu.cu"),
    ),
    (
      [
        *shlex.split(os.environ.get("NVCC", "nvcc")),
        f"-arch={gpu_arch}",
        "-O3",
        "-std=c++17",
        "--expt-extended-lambda",
        "-shared",
        "-Xcompiler=-fPIC",
        "-I",
        str(generated_dir),
        "-I",
        str(nacho_dir / "runtime"),
        "-x",
        "cu",
        str(backend_dir / f"benchmark_{operation}.cpp"),
        str(generated_dir / f"{operation}_gpu.cu"),
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


if __name__ == "__main__":
  main()
