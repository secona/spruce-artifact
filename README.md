# Artifact

```sh
git submodule update --init --recursive --depth=1
```

Run Python programs as modules from the repository root:

```sh
python -m systems.cusparse.build
python -m systems.cusparse.run --operation spmv
python -m benchmarks.plot
python -m dataset.sparse-matrix-data-helper download
```

Source, dataset, and generated output paths are relative to the repository root.
For SparseTIR, make the checkout's Python package available when running its builder:

```sh
PYTHONPATH=third_party/sparsetir/python python -m systems.sparsetir.build
```
