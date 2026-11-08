#include "common.cuh"

namespace spruce::cusparse {
using namespace detail;

namespace {
__global__ void weight(int64_t nnz, const float *a, float *d) {
  for (int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; p < nnz;
       p += int64_t(blockDim.x) * gridDim.x)
    d[p] *= a[p];
}
} // namespace

void sddmm(int m, int n, int rank, int64_t nnz, int *offsets, int *columns,
           float *a, float *b, float *c, float *d, cudaStream_t stream,
           const Benchmark *benchmark) {
  Handle handle;
  create_handle(handle, stream);
  DenseMatrix left, right;
  SparseMatrix output;
  check(cusparseCreateDnMat(&left.value, m, rank, rank, b, CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));
  // C is n x rank, so the sampled product is B * transpose(C).
  check(cusparseCreateDnMat(&right.value, n, rank, rank, c, CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));
  create_csr(output, m, n, nnz, offsets, columns, d);
  const float alpha = 1.0f, beta = 0.0f;
  size_t bytes = 0;
  check(cusparseSDDMM_bufferSize(
      handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, left.value, right.value, &beta,
      output.value, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bytes));
  Buffer buffer(bytes);
  check(cusparseSDDMM_preprocess(
      handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &alpha, left.value, right.value, &beta,
      output.value, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buffer.value));
  execute(
      [&] {
        check(cusparseSDDMM(handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_TRANSPOSE, &alpha, left.value,
                            right.value, &beta, output.value, CUDA_R_32F,
                            CUSPARSE_SDDMM_ALG_DEFAULT, buffer.value));
        if (nnz > 0) {
          int blocks = static_cast<int>(
              (nnz + 255) / 256 < 65535 ? (nnz + 255) / 256 : 65535);
          weight<<<blocks, 256, 0, stream>>>(nnz, a, d);
          check(cudaGetLastError());
        }
      },
      stream, benchmark);
  check(cudaStreamSynchronize(stream));
}

} // namespace spruce::cusparse
