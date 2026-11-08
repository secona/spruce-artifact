#include "common.cuh"

void spmv(int m, int n, int64_t nnz, int *offsets, int *columns, float *values,
          float *x, float *y, cudaStream_t stream) {
  Handle handle;
  SparseMatrix a;
  DenseVector b, c;

  cusparse_create_handle(handle, stream);
  cusparse_create_csr(a, m, n, nnz, offsets, columns, values);
  CHECK_CUSPARSE(cusparseCreateDnVec(&b.value, n, x, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&c.value, m, y, CUDA_R_32F));

  const float alpha = 1.0f, beta = 0.0f;
  size_t bytes = 0;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a.value, b.value,
      &beta, c.value, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
  Buffer buffer(bytes);
  CHECK_CUSPARSE(cusparseSpMV(
      handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a.value, b.value,
      &beta, c.value, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer.value));
  CHECK_CUDA(cudaStreamSynchronize(stream));
}
