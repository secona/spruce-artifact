#pragma once

#include "kernels.h"
#include <cusparse.h>
#include <stdexcept>

namespace spruce::cusparse::detail {

inline void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

inline void check(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(cusparseGetErrorString(status));
}

template <typename T, auto destroy>
struct Resource {
    T value = nullptr;
    Resource() = default;
    Resource(const Resource &) = delete;
    Resource &operator=(const Resource &) = delete;
    ~Resource() { if (value) destroy(value); }
};

using Handle = Resource<cusparseHandle_t, cusparseDestroy>;
using SparseMatrix = Resource<cusparseSpMatDescr_t, cusparseDestroySpMat>;
using DenseMatrix = Resource<cusparseDnMatDescr_t, cusparseDestroyDnMat>;
using DenseVector = Resource<cusparseDnVecDescr_t, cusparseDestroyDnVec>;
using Product = Resource<cusparseSpGEMMDescr_t, cusparseSpGEMM_destroyDescr>;

struct Buffer : Resource<void *, cudaFree> {
    explicit Buffer(size_t bytes) {
        // Keep a non-null pointer for cuSPARSE execution even for zero scratch.
        check(cudaMalloc(&value, bytes ? bytes : 1));
    }
    template <typename T> T *release() {
        auto result = static_cast<T *>(value);
        value = nullptr;
        return result;
    }
};

inline void create_csr(SparseMatrix &matrix, int rows, int cols, int64_t nnz,
                       int *offsets, int *columns, float *values) {
    check(cusparseCreateCsr(&matrix.value, rows, cols, nnz, offsets, columns,
                            values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
}

inline void create_handle(Handle &handle, cudaStream_t stream) {
    check(cusparseCreate(&handle.value));
    check(cusparseSetStream(handle.value, stream));
    check(cusparseSetPointerMode(handle.value, CUSPARSE_POINTER_MODE_HOST));
}

} // namespace spruce::cusparse::detail
