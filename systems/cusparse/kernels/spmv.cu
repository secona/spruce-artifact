#include "common.cuh"

namespace spruce::cusparse {
using namespace detail;

void spmv(int m, int n, int64_t nnz, int *offsets, int *columns, float *values,
          float *x, float *y, cudaStream_t stream, const Benchmark *benchmark) {
  Handle handle;
  create_handle(handle, stream);
  SparseMatrix a;
  DenseVector b, c;
  create_csr(a, m, n, nnz, offsets, columns, values);
  check(cusparseCreateDnVec(&b.value, n, x, CUDA_R_32F));
  check(cusparseCreateDnVec(&c.value, m, y, CUDA_R_32F));
  const float alpha = 1.0f, beta = 0.0f;
  size_t bytes = 0;
  check(cusparseSpMV_bufferSize(handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, a.value, b.value, &beta, c.value,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
  Buffer buffer(bytes);
  execute(
      [&] {
        check(cusparseSpMV(handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, a.value, b.value, &beta, c.value, CUDA_R_32F,
                           CUSPARSE_SPMV_ALG_DEFAULT, buffer.value));
      },
      stream, benchmark);
  check(cudaStreamSynchronize(stream));
}

} // namespace spruce::cusparse
