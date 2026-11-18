# Artifact

## GPU Operation Support

| System | SpMV | SDDMM | SpGEMM | MTTKRP |
| --- | :---: | :---: | :---: | :---: |
| cuSPARSE | ✅ | ✅ | ✅ | ✅ |
| MLIR Sparse Tensor | ✅ | ❌ | ❌ | ✅ |
| Nacho | ❌ | ❌ | ✅ | ❌ |
| SparseTIR | ✅ | ✅ | ❌ | ✅ |

## Native setup

```sh
git submodule update --init --recursive --depth=1
bash dataset/download.sh
bash dataset/extract.sh
python -m dataset.generate_inputs
```

Run these commands from the repository root.

## Docker setup

```sh
docker build -t spruce-artifact .
mkdir -p generated
docker run --rm -it --gpus all -v "$PWD/generated:/work/generated" spruce-artifact
```

## cuSPARSE

```sh
python -m systems.cusparse.build
python -m systems.cusparse.run --operation spmv
python -m systems.cusparse.run --operation sddmm
python -m systems.cusparse.run --operation spgemm
python -m systems.cusparse.run --operation mttkrp
```

## MLIR Sparse Tensor

The image sets `LLVM_DIR=/opt/llvm` for its bundled LLVM/MLIR 23.1.6.
For a compatible external installation, add
`-v "/path/to/llvm/install:/opt/llvm:ro"` before the image name in the
Docker run command. For native runs, set `LLVM_DIR` to your installation.

```sh
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
