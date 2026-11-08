#include "common.cuh"
#include <limits>

namespace spruce::cusparse {
using namespace detail;

void spgemm(int m, int k, int n, int64_t nnz_a, int *offsets_a, int *columns_a,
            float *values_a, int64_t nnz_b, int *offsets_b, int *columns_b,
            float *values_b, int *&offsets_c, int *&columns_c, float *&values_c,
            int64_t &nnz_c, cudaStream_t stream) {
  Handle handle;
  SparseMatrix a, b, c;
  Product product;

  cusparse_create_handle(handle, stream);
  cusparse_create_csr(a, m, k, nnz_a, offsets_a, columns_a, values_a);
  cusparse_create_csr(b, k, n, nnz_b, offsets_b, columns_b, values_b);

  Buffer rows((size_t(m) + 1) * sizeof(int));
  cusparse_create_csr(c, m, n, 0, static_cast<int *>(rows.value), nullptr,
                      nullptr);

  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&product.value));
  const auto op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const auto algorithm = CUSPARSE_SPGEMM_DEFAULT;
  const float alpha = 1.0f, beta = 0.0f;
  size_t bytes1 = 0, bytes2 = 0;

  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
      handle.value, op, op, &alpha, a.value, b.value, &beta, c.value,
      CUDA_R_32F, algorithm, product.value, &bytes1, nullptr));

  Buffer buffer1(bytes1);
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
      handle.value, op, op, &alpha, a.value, b.value, &beta, c.value,
      CUDA_R_32F, algorithm, product.value, &bytes1, buffer1.value));

  CHECK_CUSPARSE(cusparseSpGEMM_compute(
      handle.value, op, op, &alpha, a.value, b.value, &beta, c.value,
      CUDA_R_32F, algorithm, product.value, &bytes2, nullptr));

  Buffer buffer2(bytes2);
  CHECK_CUSPARSE(cusparseSpGEMM_compute(
      handle.value, op, op, &alpha, a.value, b.value, &beta, c.value,
      CUDA_R_32F, algorithm, product.value, &bytes2, buffer2.value));

  int64_t result_rows, result_cols, count;
  CHECK_CUSPARSE(
      cusparseSpMatGetSize(c.value, &result_rows, &result_cols, &count));

  if (count > std::numeric_limits<int>::max())
    throw std::overflow_error("SpGEMM output exceeds int32 CSR offsets");

  Buffer columns(size_t(count) * sizeof(int));
  Buffer values(size_t(count) * sizeof(float));
  CHECK_CUSPARSE(
      cusparseCsrSetPointers(c.value, rows.value, columns.value, values.value));
  CHECK_CUSPARSE(cusparseSpGEMM_copy(handle.value, op, op, &alpha, a.value,
                                     b.value, &beta, c.value, CUDA_R_32F,
                                     algorithm, product.value));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  offsets_c = rows.release<int>();
  columns_c = columns.release<int>();
  values_c = values.release<float>();
  nnz_c = count;
}

} // namespace spruce::cusparse
