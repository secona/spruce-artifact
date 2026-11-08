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

## MLIR Sparse Tensor

```sh
export LLVM_DIR=/path/to/llvm/build
python -m systems.mlir_sparse_tensor.build
python -m systems.mlir_sparse_tensor.run --operation spmv
```

The LLVM build must provide `libmlir_c_runner_utils` and `libmlir_cuda_runtime`.

## Plot Results

```sh
python -m benchmarks.plot
```
