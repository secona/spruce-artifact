# Nacho SpGEMM benchmarks

Build the compiler, generate SpGEMM, then compile and link the benchmark library:

```sh
python -m systems.nacho.build
```

`build.py` retains the original compiler helper and source-generation flow,
then compiles and links the generated kernel and benchmark bridge. It produces
`generated/nacho/libnacho_spgemm.so`. Build prerequisites are CMake, a C++20 host
compiler, and a CUDA Toolkit with CUB/Thrust and support for `cudaMallocAsync`.
Set `CXX` and `NVCC` to select compilers or supply compiler flags. Rebuild after
changing the compiler, generated kernel, runtime headers, or bridge.

Run a square matrix product `A @ A`, or supply a compatible second matrix:

```sh
python -m systems.nacho.run --matrix dataset/bcsstk01.mtx
python -m systems.nacho.run --matrix A.mtx --matrix-b B.mtx --output generated/nacho/spgemm.csv
```

`run.py` requires NumPy, SciPy, CUDA runtime libraries, and a working GPU. It loads
the prebuilt library and does no compilation or numerical verification. It
converts real Matrix Market inputs to sorted float32 CSR with int32 indices,
expanding symmetry and summing duplicates. An omitted B means A itself; A must
then be square. Explicit inputs must have compatible inner dimensions.

The bridge invokes `spgemm::spgemm_gpu_i32_f32`, the same generated function used
by the `nacho.gpu_spgemm_f32` Python binding in
`third_party/nacho/benchmarks/spgemm.py`. This path needs no Nacho Python extension
or PyTorch installation. It follows the existing benchmark defaults: 5 warmups,
14 timed repetitions, and 256 blocks with 256 threads each. Override with
`--warmup`, `--repeat`, `--blocks`, and `--threads-per-block`; select a GPU with
`--device`.

CUDA events record each generated SpGEMM call on stream 0, matching the stream
used by the generated kernel. Input loading, conversion, GPU upload, and result
release are outside the measured interval. The generated call includes its
intermediate and output allocations, partitioning, symbolic work, numeric
computation, and internal synchronization. It is a full generated-call timing,
not a numeric-only measurement with reusable output structure.

Every timed repetition is appended to `generated/nacho/spgemm.csv` by default.
Rows contain input paths, dimensions, nnz, launch geometry, warmup count, timing
scope, run timestamp, iteration, and elapsed milliseconds. Warmups are excluded.
No samples are removed from CSV. The printed trimmed mean drops the two fastest
and two slowest samples, following Nacho's timing helper; `--trim` changes that
summary only. For short runs, use e.g. `--repeat 3 --trim 0`.
