#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>

#include "../../common/cuda_utils.h"

// Device arrays, float32 values, zero-based int32 CSR indices. Dense matrices
// are row-major. Calls complete on the supplied stream before returning.
// Inputs and preallocated outputs must not overlap.
namespace spruce::cusparse {

// Optional steady-state CUDA-event measurements, in milliseconds.
// samples points to space for repetitions floats, owned by the host caller.
struct Benchmark {
  int warmup;
  int repetitions;
  float *samples;
};

// y[i] = sum_j A[i,j] * x[j]. Caller allocates y (m elements).
void spmv(int m, int n, int64_t nnz, int *offsets, int *columns, float *values,
          float *x, float *y, cudaStream_t stream = nullptr,
          const Benchmark *benchmark = nullptr);

// d[p] = a[p] * dot(B[i,:], C[j,:]) for CSR entry p at (i,j).
// Caller allocates d (nnz elements); output uses the input CSR structure.
void sddmm(int m, int n, int rank, int64_t nnz, int *offsets, int *columns,
           float *a, float *b, float *c, float *d,
           cudaStream_t stream = nullptr, const Benchmark *benchmark = nullptr);

// C = A (m x k) * B (k x n). Output references are host variables.
// On success, caller owns the three output arrays and frees them with cudaFree.
void spgemm(int m, int k, int n, int64_t nnz_a, int *offsets_a, int *columns_a,
            float *values_a, int64_t nnz_b, int *offsets_b, int *columns_b,
            float *values_b, int *&offsets_c, int *&columns_c, float *&values_c,
            int64_t &nnz_c, cudaStream_t stream = nullptr);

} // namespace spruce::cusparse

namespace spruce::cusparse::detail {

template <typename T, auto destroy> struct Resource {
  T value = nullptr;
  Resource() = default;
  Resource(const Resource &) = delete;
  Resource &operator=(const Resource &) = delete;
  ~Resource() {
    if (value)
      destroy(value);
  }
};

using Handle = Resource<cusparseHandle_t, cusparseDestroy>;
using SparseMatrix = Resource<cusparseSpMatDescr_t, cusparseDestroySpMat>;
using DenseMatrix = Resource<cusparseDnMatDescr_t, cusparseDestroyDnMat>;
using DenseVector = Resource<cusparseDnVecDescr_t, cusparseDestroyDnVec>;
using Product = Resource<cusparseSpGEMMDescr_t, cusparseSpGEMM_destroyDescr>;
using Event = Resource<cudaEvent_t, cudaEventDestroy>;

template <typename Function>
void execute(Function launch, cudaStream_t stream, const Benchmark *benchmark) {
  if (!benchmark) {
    launch();
    return;
  }
  if (benchmark->warmup < 0 || benchmark->repetitions < 1 ||
      !benchmark->samples)
    throw std::invalid_argument("Invalid benchmark configuration");
  Event start, stop;
  CHECK_CUDA(cudaEventCreate(&start.value));
  CHECK_CUDA(cudaEventCreate(&stop.value));
  for (int i = 0; i < benchmark->warmup; ++i)
    launch();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (int i = 0; i < benchmark->repetitions; ++i) {
    CHECK_CUDA(cudaEventRecord(start.value, stream));
    launch();
    CHECK_CUDA(cudaEventRecord(stop.value, stream));
    CHECK_CUDA(cudaEventSynchronize(stop.value));
    CHECK_CUDA(
        cudaEventElapsedTime(&benchmark->samples[i], start.value, stop.value));
  }
}

struct Buffer : Resource<void *, cudaFree> {
  explicit Buffer(size_t bytes) {
    // Keep a non-null pointer for cuSPARSE execution even for zero scratch.
    CHECK_CUDA(cudaMalloc(&value, bytes ? bytes : 1));
  }
  template <typename T> T *release() {
    auto result = static_cast<T *>(value);
    value = nullptr;
    return result;
  }
};

inline void create_csr(SparseMatrix &matrix, int rows, int cols, int64_t nnz,
                       int *offsets, int *columns, float *values) {
  CHECK_CUSPARSE(cusparseCreateCsr(&matrix.value, rows, cols, nnz, offsets,
                                   columns, values, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
}

inline void create_handle(Handle &handle, cudaStream_t stream) {
  CHECK_CUSPARSE(cusparseCreate(&handle.value));
  CHECK_CUSPARSE(cusparseSetStream(handle.value, stream));
  CHECK_CUSPARSE(
      cusparseSetPointerMode(handle.value, CUSPARSE_POINTER_MODE_HOST));
}

} // namespace spruce::cusparse::detail
