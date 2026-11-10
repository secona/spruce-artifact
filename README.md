# Artifact

## GPU Operation Support

| System | SpMV | SDDMM | SpGEMM | MTTKRP |
| --- | :---: | :---: | :---: | :---: |
| cuSPARSE | ✅ | ✅ | ✅ | ✅ |
| MLIR Sparse Tensor | ✅ | ❌ | ❌ | ✅ |
| Nacho | ❌ | ❌ | ✅ | ❌ |
| SparseTIR | ✅ | ✅ | ❌ | ✅ |

## Setup

```sh
git submodule update --init --recursive --depth=1
bash dataset/download.sh
bash dataset/extract.sh
python -m dataset.generate_inputs
```

Run these commands from the repository root.

## cuSPARSE

```sh
python -m systems.cusparse.build
python -m systems.cusparse.run --operation spmv
python -m systems.cusparse.run --operation sddmm
python -m systems.cusparse.run --operation spgemm
python -m systems.cusparse.run --operation mttkrp
```

The cuSPARSE MTTKRP baseline unfolds the tensor into CSR with one column per
observed `(j, k)` pair. Each timed iteration constructs the corresponding rows
of the Khatri–Rao product on the GPU and multiplies with `cusparseSpMM`.
Tensor conversion and host-to-device input copies happen before timing.

## MLIR Sparse Tensor

```sh
export LLVM_DIR=/path/to/llvm/install
python -m systems.mlir_sparse_tensor.build
python -m systems.mlir_sparse_tensor.run --operation spmv
python -m systems.mlir_sparse_tensor.run --operation mttkrp
```

The LLVM installation must provide the MLIR headers, `libmlir_c_runner_utils`,
and `libmlir_cuda_runtime`.

To use an uninstalled LLVM build, set `LLVM_DIR` to its build directory and
`MLIR_INCLUDE_DIR` to `/path/to/llvm-project/mlir/include` for the source headers.

## SparseTIR

```sh
bash build_deps.sh sparsetir
export PYTHONPATH="$PWD/third_party/sparsetir/python${PYTHONPATH:+:$PYTHONPATH}"
python -m systems.sparsetir.build
python -m systems.sparsetir.run --operation spmv
python -m systems.sparsetir.run --operation sddmm
python -m systems.sparsetir.run --operation mttkrp
```

## Plot Results

```sh
python -m benchmarks.plot
```
