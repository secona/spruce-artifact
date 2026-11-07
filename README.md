# Artifact

## GPU Operation Support

| System | SpMV | SDDMM | SpGEMM | MTTKRP |
| --- | :---: | :---: | :---: | :---: |
| cuSPARSE | ✅ | ✅ | ✅ | ❌ |
| Nacho | ❌ | ❌ | ✅ | ❌ |
| SparseTIR | ✅ | ✅ | ❌ | ⏸️ |
| MLIR Sparse Tensor | ✅ | ❌ | ❌ | ⏸️ |

## Setup

```sh
git submodule update --init --recursive --depth=1
python -m dataset.sparse-matrix-data-helper download
```

## cuSPARSE

```sh
python -m systems.cusparse.build
python -m systems.cusparse.run --operation spmv
python -m systems.cusparse.run --operation sddmm
python -m systems.cusparse.run --operation spgemm
```

## Nacho

```sh
python -m systems.nacho.build
python -m systems.nacho.run --operation spgemm
```

## SparseTIR

```sh
export LLVM_14_BUILD_DIR=/path/to/llvm14/build/dir
./build_deps.sh sparsetir
PYTHONPATH=third_party/sparsetir/python python -m systems.sparsetir.build
PYTHONPATH=third_party/sparsetir/python python -m systems.sparsetir.run --operation spmv
PYTHONPATH=third_party/sparsetir/python python -m systems.sparsetir.run --operation sddmm
```

The runner compiles kernels for each case's dimensions before timing them with CUDA events.
It reads the SpMV/SDDMM configs and appends samples to `generated/sparsetir/<operation>.csv`.

## MLIR Sparse Tensor

```sh
export LLVM_DIR=/path/to/llvm/build
python -m systems.mlir_sparse_tensor.build
python -m systems.mlir_sparse_tensor.run --operation spmv
```

The SpMV runner reads `benchmarks/spmv/config.py` and writes
`generated/mlir_sparse_tensor/spmv.csv`. Its wall-clock samples include the generated
call's allocations and host/device transfers, recorded in the CSV timing and scope fields.
The LLVM build must provide `libmlir_c_runner_utils` and `libmlir_cuda_runtime`.

## Plot Results

```sh
python -m benchmarks.plot
```
