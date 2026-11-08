#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>

#include "../../common/cuda_utils.h"

// Device arrays, float32 values, zero-based int32 CSR indices. Dense matrices
// are row-major. Calls complete on the supplied stream before returning.
// Inputs and preallocated outputs must not overlap.

// Whole-call CUDA-event measurements used by the benchmark wrappers, in
// milliseconds. samples points to space for repetitions floats, owned by the
// host caller.
struct Benchmark {
  int warmup;
  int repetitions;
  float *samples;
};

using Handle = Resource<cusparseHandle_t, cusparseDestroy>;
using SparseMatrix = Resource<cusparseSpMatDescr_t, cusparseDestroySpMat>;
using DenseMatrix = Resource<cusparseDnMatDescr_t, cusparseDestroyDnMat>;
using DenseVector = Resource<cusparseDnVecDescr_t, cusparseDestroyDnVec>;
using Product = Resource<cusparseSpGEMMDescr_t, cusparseSpGEMM_destroyDescr>;

template <typename Function>
void execute(Function launch, cudaStream_t stream, const Benchmark *benchmark) {
  if (!benchmark) {
    launch();
    return;
  }

  if (benchmark->warmup < 0 || benchmark->repetitions < 1 ||
      !benchmark->samples)
    throw std::invalid_argument("Invalid benchmark configuration");

  CudaEvent start, stop;

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

inline void cusparse_create_csr(SparseMatrix &matrix, int rows, int cols,
                                int64_t nnz, int *offsets, int *columns,
                                float *values) {
  CHECK_CUSPARSE(cusparseCreateCsr(&matrix.value, rows, cols, nnz, offsets,
                                   columns, values, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));
}

inline void cusparse_create_handle(Handle &handle, cudaStream_t stream) {
  CHECK_CUSPARSE(cusparseCreate(&handle.value));
  CHECK_CUSPARSE(cusparseSetStream(handle.value, stream));
  CHECK_CUSPARSE(
      cusparseSetPointerMode(handle.value, CUSPARSE_POINTER_MODE_HOST));
}
