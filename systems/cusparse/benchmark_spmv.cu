// C ABI for the SpMV benchmark runner. Host arrays are copied before timing.
#include "common.cuh"
#include <string>

using namespace spruce::cusparse;
using namespace spruce::cusparse::detail;

extern "C" const char *
spruce_benchmark_spmv(int device, int m, int n, int64_t nnz, const int *offsets,
                      const int *columns, const float *values, const float *x,
                      int warmup, int repetitions, float *samples) {
  static thread_local std::string error;
  try {
    CHECK_CUDA(cudaSetDevice(device));
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking));
    Buffer rows((size_t(m) + 1) * sizeof(int));
    Buffer cols(size_t(nnz) * sizeof(int)), data(size_t(nnz) * sizeof(float));
    Buffer input(size_t(n) * sizeof(float));
    CHECK_CUDA(cudaMemcpy(rows.value, offsets, (size_t(m) + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (nnz) {
      CHECK_CUDA(cudaMemcpy(cols.value, columns, size_t(nnz) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(data.value, values, size_t(nnz) * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMemcpy(input.value, x, size_t(n) * sizeof(float),
                          cudaMemcpyHostToDevice));
    Benchmark benchmark{warmup, repetitions, samples};
    execute(
        [&] {
          Buffer output(size_t(m) * sizeof(float));
          spmv(m, n, nnz, static_cast<int *>(rows.value),
               static_cast<int *>(cols.value), static_cast<float *>(data.value),
               static_cast<float *>(input.value),
               static_cast<float *>(output.value), stream.value);
        },
        stream.value, &benchmark);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
