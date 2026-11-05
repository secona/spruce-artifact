| Program | Formats | Implementation |
| --- | --- | --- |
| SpMV | CSR matrix, dense vector and output | `cusparseSpMV` |
| SpGEMM | CSR inputs and output | `cusparseSpGEMM` with output allocation |
| SDDMM | CSR sampling matrix and output, row-major dense factors | `cusparseSDDMM` followed by multiplication by the sampling values |
| MTTKRP | CSF tensor, dense factors and output | Unsupported: no direct cuSPARSE tensor operation |

Emit wrappers into `generated/cusparse`, following the other backend build scripts:

```sh
python backends/cusparse/build.py
python backends/cusparse/build.py --operation spmv
```

cuSPARSE ships precompiled kernels with the CUDA Toolkit, so this step copies
the wrappers and headers without invoking a compiler or requiring a GPU.
To compile the emitted sources (CUDA Toolkit with cuSPARSE and a C++17 host
compiler required):

```sh
nvcc -std=c++17 -shared -Xcompiler=-fPIC generated/cusparse/*.cu -lcusparse -o generated/cusparse/libcusparse_backend.so
```

The C++ entry points are declared in `kernels.h` in namespace `spruce::cusparse`.
All tensor arrays reside on the GPU and use float32 values and zero-based int32
CSR indices. Dense factors use row-major storage. Inputs and outputs must not
overlap. SpMV and SDDMM write caller-allocated outputs; SpGEMM allocates its
output arrays, which the caller releases with `cudaFree`. Calls create their
own handle, descriptors, and workspace and synchronize the supplied stream
before returning. CUDA/cuSPARSE failures throw C++ exceptions. Timings therefore
include setup, allocation, and synchronization, rather than only kernel execution.

SDDMM computes `D[i,j] = A[i,j] * sum_k B[i,k] * C[j,k]`, matching the
SparseTIR and Nacho definitions. cuSPARSE's sampled product uses only the CSR
structure; the final CUDA kernel applies the sampling values.

API reference: [NVIDIA cuSPARSE documentation](https://docs.nvidia.com/cuda/cusparse/).
