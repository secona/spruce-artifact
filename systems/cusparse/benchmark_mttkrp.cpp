// C ABI for MTTKRP via a compact Khatri-Rao product and cuSPARSE SpMM.
// Host unfolding and input copies happen before timing; the product is timed.
#include "kernels/common.h"
#include "kernels/kernels.h"
#include <string>

extern "C" const char *spruce_benchmark_mttkrp(
    int device, int m, int n, int p, int rank, int64_t nnz, int pairs,
    const int *offsets, const int *columns, const float *values,
    const int *pair_j, const int *pair_k, const float *b, const float *c,
    int warmup, int repetitions, float *samples, float *result) {
  static thread_local std::string error;
  try {
    CHECK_CUDA(cudaSetDevice(device));
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking));
    Buffer rows((size_t(m) + 1) * sizeof(int));
    Buffer cols(size_t(nnz) * sizeof(int)), data(size_t(nnz) * sizeof(float));
    Buffer indices_j(size_t(pairs) * sizeof(int));
    Buffer indices_k(size_t(pairs) * sizeof(int));
    Buffer left(size_t(n) * rank * sizeof(float));
    Buffer right(size_t(p) * rank * sizeof(float));
    CHECK_CUDA(cudaMemcpy(rows.value, offsets, (size_t(m) + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cols.value, columns, size_t(nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data.value, values, size_t(nnz) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(indices_j.value, pair_j, size_t(pairs) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(indices_k.value, pair_k, size_t(pairs) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(left.value, b, size_t(n) * rank * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(right.value, c, size_t(p) * rank * sizeof(float),
                          cudaMemcpyHostToDevice));
    Benchmark benchmark{warmup, repetitions, samples};
    execute(
        [&] {
          Buffer output(size_t(m) * rank * sizeof(float));
          mttkrp(m, rank, nnz, pairs, static_cast<int *>(rows.value),
                  static_cast<int *>(cols.value),
                  static_cast<float *>(data.value),
                  static_cast<int *>(indices_j.value),
                  static_cast<int *>(indices_k.value),
                  static_cast<float *>(left.value),
                  static_cast<float *>(right.value),
                  static_cast<float *>(output.value), stream.value);
          if (result)
            CHECK_CUDA(cudaMemcpy(result, output.value,
                                  size_t(m) * rank * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        },
        stream.value, samples ? &benchmark : nullptr);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
