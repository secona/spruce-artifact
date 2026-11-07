| Program | Formats | Implementation |
| --- | --- | --- |
| SpMV | CSR matrix, dense vector and output | `cusparseSpMV` |
| SpGEMM | CSR inputs and output | `cusparseSpGEMM` with output allocation |
| SDDMM | CSR sampling matrix and output, row-major dense factors | `cusparseSDDMM` followed by multiplication by the sampling values |
| MTTKRP | CSF tensor, dense factors and output | Unsupported: no direct cuSPARSE tensor operation |

Build the benchmark libraries (CUDA Toolkit with cuSPARSE and a C++17
host compiler required):

```sh
python -m systems.cusparse.build
```

`build()` compiles and links each kernel with its benchmark bridge into
`generated/cusparse/libcusparse_<operation>.so`. Set `NVCC` to select the compiler.
Compilation does not require a GPU. Rebuild after changing native sources.
The benchmark workflow supports SpMV, SDDMM, and SpGEMM.

The C++ entry points are declared in `kernels.h` in namespace `spruce::cusparse`.
All tensor arrays reside on the GPU and use float32 values and zero-based int32
CSR indices. Dense factors use row-major storage. Inputs and outputs must not
overlap. SpMV and SDDMM write caller-allocated outputs; SpGEMM allocates its
output arrays, which the caller releases with `cudaFree`. Calls create their
own handle, descriptors, and workspace and synchronize the supplied stream
before returning. CUDA/cuSPARSE failures throw C++ exceptions. Externally timing
ordinary calls therefore includes setup, allocation, and synchronization.
The optional benchmark mode described below measures execution after setup.

SDDMM computes `D[i,j] = A[i,j] * sum_k B[i,k] * C[j,k]`, matching the
SparseTIR and Nacho definitions. cuSPARSE's sampled product uses only the CSR
structure; the final CUDA kernel applies the sampling values.

### CUDA-event benchmarks

The runner requires Python with NumPy and SciPy, the CUDA runtime libraries,
and a working CUDA GPU. It loads the prebuilt library and benchmarks `y = A @ x`.

```sh
python -m systems.cusparse.build
python -m systems.cusparse.run --operation spmv
python -m systems.cusparse.run --operation spmv --warmup 10 --repeat 100
python -m dataset.sparse-matrix-data-helper download
python -m systems.cusparse.run --operation sddmm
```

The Matrix Market input is converted to sorted CSR with float32 values and
zero-based int32 indices. Symmetry is expanded and duplicates are summed.
The dataset helper generates the dense vector and SDDMM factors using seed 1234.
SDDMM factors are float32 arrays with shapes `(m, 32)` and `(n, 32)`.
Select a GPU with `--device` (default 0).

Handles, descriptors, workspace, and input uploads are prepared once. After
warmup, each repetition records start and stop CUDA events around `cusparseSpMV`
on the same stream, waits for the stop event, and obtains elapsed milliseconds.
Setup, allocation, transfers, and warmup are outside the timed region.
SDDMM likewise prepares descriptors, workspace, and preprocessing once, then
times `cusparseSDDMM` and the sampling-value multiplication together.
CUDA events measure elapsed stream time; host launch gaps can contribute to it.

The runner prints median, mean, and minimum execution time. Each timed repetition
is appended as a CSV row to `generated/cusparse/<operation>.csv`.
Rows include a UTC run timestamp, input paths and dimensions,
nnz, device, warmup count, iteration number (starting at 1), and `elapsed_ms`.
Repeated benchmark invocations append to the same file with one header; warmup
iterations are excluded.
It performs no numerical verification and does not save the computed vector.

API reference: [NVIDIA cuSPARSE documentation](https://docs.nvidia.com/cuda/cusparse/).
