# Artifact

## GPU Operation Support

| System | SpMV | SDDMM | SpGEMM | MTTKRP |
| --- | :---: | :---: | :---: | :---: |
| cuSPARSE | ✅ | ✅ | ✅ | ❌ |
| Nacho | ❌ | ❌ | ✅ | ❌ |
| SparseTIR | ⏸️ | ⏸️ | ❌ | ⏸️ |
| MLIR Sparse Tensor | ⏸️ | ❌ | ❌ | ⏸️ |

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
```

## MLIR Sparse Tensor

```sh
export LLVM_DIR=/path/to/llvm/build
python -m systems.mlir_sparse_tensor.build
```

## Plot Results

```sh
python -m benchmarks.plot
```
