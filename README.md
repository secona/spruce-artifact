# Artifact

## Submodules

```sh
git submodule update --init --recursive --depth=1
```

## Building

These are the environment variables required to build the artifact.

- `LLVM_14_BUILD_DIR` points to the `build` directory of an LLVM 14 checkout.

## Running Python Programs

Run Python programs as modules from the repository root:

```sh
python -m systems.cusparse.build
python -m systems.cusparse.run --operation spmv
python -m systems.cusparse.run --operation sddmm
python -m benchmarks.plot
python -m dataset.sparse-matrix-data-helper download
```

Source, dataset, and generated output paths are relative to the repository root.
Run the dataset downloader before SDDMM to generate its reproducible float32
dense factors (rank 32). SDDMM records CUDA-event samples, including the sampling
value multiplication, in `generated/cusparse/sddmm.csv`.
`./build_deps.sh sparsetir` applies `patches/sparsetir-numpy-2.patch` directly
to the SparseTIR submodule before building, skipping it if already applied.
The patch removes the obsolete `np.float_` alias; `np.float64` is already mapped.
No fork or submodule commit is needed. To apply it manually from the repository root:

```sh
git -C third_party/sparsetir apply ../../patches/sparsetir-numpy-2.patch
```

For SparseTIR, make the checkout's Python package available when running its builder:

```sh
PYTHONPATH=third_party/sparsetir/python python -m systems.sparsetir.build
```
