#include "common.h"
#include "kernels.h"

namespace {
__global__ void khatri_rao(int pairs, int rank, const int *pair_j,
                          const int *pair_k, const float *b, const float *c,
                          float *product) {
  const size_t count = size_t(pairs) * rank;
  for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += size_t(blockDim.x) * gridDim.x) {
    const size_t pair = index / rank;
    const int r = index % rank;
    product[index] = b[size_t(pair_j[pair]) * rank + r] *
                     c[size_t(pair_k[pair]) * rank + r];
  }
}
} // namespace

void mttkrp(int m, int rank, int64_t nnz, int pairs, int *offsets,
            int *columns, float *values, int *pair_j, int *pair_k, float *b,
            float *c, float *d, cudaStream_t stream) {
  Handle handle;
  SparseMatrix a;
  DenseMatrix h, output;
  Buffer product(size_t(pairs) * rank * sizeof(float));

  // Compute only Khatri-Rao rows referenced by the sparse unfolding.
  const size_t blocks = (size_t(pairs) * rank + 255) / 256;
  khatri_rao<<<static_cast<unsigned>(blocks < 65535 ? blocks : 65535), 256,
               0, stream>>>(pairs, rank, pair_j, pair_k, b, c,
                            static_cast<float *>(product.value));
  CHECK_CUDA(cudaGetLastError());

  cusparse_create_handle(handle, stream);
  cusparse_create_csr(a, m, pairs, nnz, offsets, columns, values);
  CHECK_CUSPARSE(cusparseCreateDnMat(&h.value, pairs, rank, rank, product.value,
                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseCreateDnMat(&output.value, m, rank, rank, d,
                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));

  const float alpha = 1.0f, beta = 0.0f;
  const auto op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t bytes = 0;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle.value, op, op, &alpha, a.value, h.value, &beta, output.value,
      CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
  Buffer buffer(bytes);
  CHECK_CUSPARSE(cusparseSpMM(
      handle.value, op, op, &alpha, a.value, h.value, &beta, output.value,
      CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer.value));
  CHECK_CUDA(cudaStreamSynchronize(stream));
}
