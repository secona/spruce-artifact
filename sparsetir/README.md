| Program | Formats | GPU codegen |
| --- | --- | --- |
| SpMV | CSR matrix, dense vector and output | Enabled; each CUDA thread owns a row |
| SpGEMM | CSR inputs and output | Skipped: constructing a new CSR output requires a separate symbolic/allocation phase; these kernels use preallocated formats |
| SDDMM | CSR weights and output, dense factors | Enabled; reuses the input CSR structure |
| MTTKRP | CSF tensor with a dense outer level, dense factors and output | Enabled; each CUDA thread owns an output row |

These are kernel build and code-generation experiments.
They use float32 values and int32 indices. SDDMM includes the CSR matrix's values
as weights: `D[i,j] = A[i,j] * sum_k B[i,k] * C[j,k]`.
MTTKRP computes `D[i,r] = sum_jk A[i,j,k] * B[j,r] * C[k,r]`.
Its outer level is dense, unlike the fully compressed tensor in `mlir/mttkrp.mlir`.
The CUDA schedules keep sparse traversal and reductions on the GPU and assign
each row to one thread. They are baseline schedules, without autotuning.

Build the pinned [SparseTIR fork of TVM](https://github.com/uwsampl/SparseTIR)
from the repository root. Use uv, Python 3.9–3.11, CMake 3.18+, a C++ compiler,
and CUDA Toolkit 11+. The default compiler build also enables LLVM; use a
compatible version such as LLVM 14, or disable it as shown below:

```sh
make submodules
uv sync --project sparsetir
make build-sparsetir
```

The experiment Makefile uses `uv run --locked` with the project's `uv.lock` and
Python 3.11 environment. It sets `PYTHONPATH` and `TVM_LIBRARY_PATH` to the submodule,
so installing another TVM package is unnecessary. To select an LLVM installation
or CUDA path, override the CMake flags:

```sh
make build-sparsetir SPARSETIR_CMAKE_FLAGS='-DUSE_CUDA=ON -DUSE_LLVM=/path/to/llvm-build/bin/llvm-config'
```

These experiments use C code generation for the host wrapper, so LLVM can be
disabled while still compiling the CUDA device kernels:

```sh
make build-sparsetir SPARSETIR_CMAKE_FLAGS='-DUSE_CUDA=ON -DUSE_LLVM=OFF'
```

Generate SparseTIR, lowered TIR, and CUDA sources in `sparsetir/generated/`:

```sh
make -C sparsetir
make -C sparsetir ARGS='--workload sddmm'
```

`experiments.py` specializes small example shapes, lowers the sparse iterations
and buffers, and calls `tvm.build` for each kernel. It saves the original SparseTIR
(`*.sparse.tir`), lowered TIR (`*.tir`), and CUDA source (`*.cu`). It does not
allocate data buffers, launch kernels, or perform timing.

The target is fixed to CUDA (`sm_80`), the feature rank is 32, and the output
directory is `sparsetir/generated/`. The explicit CUDA architecture allows
compilation without querying a GPU. Building needs a CUDA-enabled SparseTIR
library and the toolkit, including `nvcc`.

You can also run the script directly with uv:

```sh
cd sparsetir
export PYTHONPATH=../third_party/sparsetir/python
export TVM_LIBRARY_PATH=../third_party/sparsetir/build
uv run --locked experiments.py --workload spmv
```

Use `UV=/path/to/uv`, `PYTHON=/path/to/python`, or
`SPARSETIR_DIR=/path/to/SparseTIR` to select another environment or build.
`make -C sparsetir clean` removes generated outputs.
