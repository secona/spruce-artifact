import argparse
from pathlib import Path


def build(operation, gpu_arch):
    import tvm
    from tvm.sparse import lower_sparse_buffer, lower_sparse_iter

    from .kernels.mttkrp import mttkrp
    from .kernels.sddmm import sddmm
    from .kernels.spmv import spmv

    if operation == "spmv":
        kernel = spmv(1024, 1024, 4096)
    elif operation == "sddmm":
        kernel = sddmm(1024, 1024, 32, 4096)
    elif operation == "mttkrp":
        kernel = mttkrp(1024, 1024, 1024, 32, 4096, 16384)
    else:
        raise ValueError(f"Unknown operation: {operation}")

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

    return output_dir / f"{operation}.cu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("spmv", "sddmm", "mttkrp"))
    parser.add_argument("--gpu-arch", default="sm_86")
    args = parser.parse_args()

    for operation in (args.operation,) if args.operation else ("spmv", "sddmm", "mttkrp"):
        print(build(operation, args.gpu_arch))
