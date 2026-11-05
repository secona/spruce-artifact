#!/usr/bin/env python3
"""Build SparseTIR kernels and save their IR and generated source code."""

import argparse
from pathlib import Path


def workload(name):
    """Specialize example shapes without allocating input or output buffers."""
    from spmv import spmv
    from sddmm import sddmm
    from mttkrp import mttkrp

    m, n, p = 4, 5, 6
    rank = 32
    if name == "spmv":
        return spmv(m, n, nnz=5)
    if name == "sddmm":
        return sddmm(m, n, rank, nnz=5)
    if name == "mttkrp":
        return mttkrp(m, n, p, rank, nnz_j=5, nnz_k=7)
    raise ValueError(f"Unknown workload: {name}")


def lower(func, name):
    import tvm
    from tvm.sparse import lower_sparse_iter, lower_sparse_buffer

    mod = lower_sparse_iter(tvm.IRModule.from_expr(func))
    sch = tvm.tir.Schedule(mod)
    block = sch.get_block(name + "0", func_name=name)
    # Each thread owns a row, so sparse reductions need no atomics.
    row = sch.get_loops(block)[0]
    blocks, threads = sch.split(row, factors=[None, 128])
    sch.bind(blocks, "blockIdx.x")
    sch.bind(threads, "threadIdx.x")
    return lower_sparse_buffer(sch.mod)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("all", "spmv", "sddmm", "mttkrp"), default="all")
    args = parser.parse_args()

    import tvm

    # An explicit architecture lets nvcc build kernels without querying a GPU.
    # A C host wrapper keeps this build-only path independent of LLVM.
    target = tvm.target.Target("cuda -arch=sm_80", host="c")
    output_dir = Path(__file__).resolve().parent / "generated"
    if tvm.get_global_func("target.build.cuda", allow_missing=True) is None:
        parser.error("SparseTIR was built without CUDA support; rebuild with -DUSE_CUDA=ON")

    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("spmv", "sddmm", "mttkrp") if args.workload == "all" else (args.workload,)
    for name in names:
        func = workload(name)
        mod = lower(func, name)
        (output_dir / f"{name}.sparse.tir").write_text(func.script() + "\n")
        (output_dir / f"{name}.tir").write_text(mod.script() + "\n")
        compiled = tvm.build(mod, target=target)
        source = compiled.imported_modules[0].get_source("cu")
        (output_dir / f"{name}.cu").write_text(source)
        print(f"Built {name} ({target}) in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
