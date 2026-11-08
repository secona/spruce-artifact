import os
from pathlib import Path
import shlex
import subprocess
from benchmarks.args import parse_build_args


OPERATIONS = ("spmv", "sddmm", "mttkrp")


def main():
  args = parse_build_args(OPERATIONS)

  for operation in (args.operation,) if args.operation else OPERATIONS:
    build(operation, args.gpu_arch)


def compile_kernel(operation, gpu_arch):
  import tvm
  from tvm.sparse import lower_sparse_buffer, lower_sparse_iter

  from .kernels.mttkrp import mttkrp
  from .kernels.sddmm import sddmm
  from .kernels.spmv import spmv

  kernels = {
    "spmv": (spmv, ("m", "n", "nnz")),
    "sddmm": (sddmm, ("m", "n", "rank", "nnz")),
    "mttkrp": (mttkrp, ("m", "n", "p", "rank", "nnz_j", "nnz_k")),
  }
  factory, dimension_names = kernels[operation]
  dimensions = [tvm.tir.Var(name, "int32") for name in dimension_names]
  kernel = factory(*dimensions)
  # Keep the existing kernel factories, exposing their symbolic dimensions as
  # scalar arguments so the exported module works for every benchmark input.
  kernel = tvm.tir.PrimFunc(
    list(kernel.params) + dimensions,
    kernel.body,
    kernel.ret_type,
    kernel.buffer_map,
    kernel.preflattened_buffer_map,
    kernel.sp_axes,
    kernel.attrs,
    kernel.span,
  )

  sparse_mod = tvm.IRModule.from_expr(kernel)
  schedule = tvm.tir.Schedule(lower_sparse_iter(sparse_mod))
  # Each thread owns a row; sparse reductions remain serial within that row.
  outer_block = schedule.get_block(f"{operation}0", func_name=operation)
  row = schedule.get_loops(outer_block)[0]
  block, thread = schedule.split(row, factors=[None, 128])
  schedule.bind(block, "blockIdx.x")
  schedule.bind(thread, "threadIdx.x")
  mod = lower_sparse_buffer(schedule.mod)

  compiled = tvm.build(mod, target=f"cuda -arch={gpu_arch}")
  cuda_source = compiled.imported_modules[0].get_source("cu")

  output_dir = Path("generated/sparsetir")
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / f"{operation}.sparse_tir.py").write_text(sparse_mod.script())
  (output_dir / f"{operation}.tir.py").write_text(mod.script())
  (output_dir / f"{operation}.cu").write_text(cuda_source)

  return compiled


def build(operation, gpu_arch):
  from tvm._ffi.libinfo import find_include_path, find_lib_path

  assert operation in OPERATIONS

  backend_dir = Path("systems/sparsetir")
  generated_dir = Path("generated") / "sparsetir"
  generated_dir.mkdir(parents=True, exist_ok=True)
  module = generated_dir / f"{operation}_kernel.so"
  library = generated_dir / f"libsparsetir_{operation}.so"
  cuda_root = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
  runtime_dir = Path(find_lib_path("libtvm_runtime.so")[0]).parent
  compiler = shlex.split(os.environ.get("CXX", "c++"))

  print(f"lowering: {backend_dir / 'kernels' / f'{operation}.py'}", flush=True)
  compiled = compile_kernel(operation, gpu_arch)

  def export_module(output, objects):
    command = [*compiler, "-shared", "-fPIC", *objects, "-o", output]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

  compiled.export_library(str(module), fcompile=export_module)
  print("built:", module)

  commands = [
    (
      [
        *compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *[flag for path in find_include_path() for flag in ("-I", path)],
        "-I",
        str(cuda_root / "include"),
        str(backend_dir / f"benchmark_{operation}.cpp"),
        "-L",
        str(runtime_dir),
        f"-Wl,-rpath,{runtime_dir}",
        "-ltvm_runtime",
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


if __name__ == "__main__":
  main()
