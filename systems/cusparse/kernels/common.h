#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>

#include "../../common/cuda_utils.h"
#include "../../common/execution.h"

using Handle = Resource<cusparseHandle_t, cusparseDestroy>;
using SparseMatrix = Resource<cusparseSpMatDescr_t, cusparseDestroySpMat>;
using DenseMatrix = Resource<cusparseDnMatDescr_t, cusparseDestroyDnMat>;
using DenseVector = Resource<cusparseDnVecDescr_t, cusparseDestroyDnVec>;
using Product = Resource<cusparseSpGEMMDescr_t, cusparseSpGEMM_destroyDescr>;

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
