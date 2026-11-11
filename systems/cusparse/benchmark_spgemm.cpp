// C ABI for the SpGEMM benchmark runner. Host arrays are copied before timing.
#include "kernels/common.h"
#include "kernels/kernels.h"
#include <memory>
#include <string>

// Host CSR returned for reference generation. Free with spruce_free_spgemm_output.
struct SpgemmOutput {
  int64_t nnz = 0;
  int *offsets = nullptr;
  int *columns = nullptr;
  float *values = nullptr;
};

extern "C" void spruce_free_spgemm_output(SpgemmOutput *output) {
  delete[] output->offsets;
  delete[] output->columns;
  delete[] output->values;
  *output = SpgemmOutput{};
}

extern "C" const char *spruce_benchmark_spgemm(
    int device, int m, int k, int n, int64_t nnz_a, const int *offsets_a,
    const int *columns_a, const float *values_a, int64_t nnz_b,
    const int *offsets_b, const int *columns_b, const float *values_b,
    int warmup, int repetitions, float *samples, SpgemmOutput *output) {
  static thread_local std::string error;
  try {
    CHECK_CUDA(cudaSetDevice(device));
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking));
    Buffer rows_a((size_t(m) + 1) * sizeof(int));
    Buffer cols_a(size_t(nnz_a) * sizeof(int)),
        data_a(size_t(nnz_a) * sizeof(float));
    Buffer rows_b((size_t(k) + 1) * sizeof(int));
    Buffer cols_b(size_t(nnz_b) * sizeof(int)),
        data_b(size_t(nnz_b) * sizeof(float));
    CHECK_CUDA(cudaMemcpy(rows_a.value, offsets_a,
                          (size_t(m) + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rows_b.value, offsets_b,
                          (size_t(k) + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (nnz_a) {
      CHECK_CUDA(cudaMemcpy(cols_a.value, columns_a,
                            size_t(nnz_a) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(data_a.value, values_a,
                            size_t(nnz_a) * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    if (nnz_b) {
      CHECK_CUDA(cudaMemcpy(cols_b.value, columns_b,
                            size_t(nnz_b) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(data_b.value, values_b,
                            size_t(nnz_b) * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    Benchmark benchmark{warmup, repetitions, samples};
    execute(
        [&] {
          Resource<int *, cudaFree> rows_c, cols_c;
          Resource<float *, cudaFree> data_c;
          int64_t nnz_c = 0;
          spgemm(m, k, n, nnz_a, static_cast<int *>(rows_a.value),
                 static_cast<int *>(cols_a.value),
                 static_cast<float *>(data_a.value), nnz_b,
                 static_cast<int *>(rows_b.value),
                 static_cast<int *>(cols_b.value),
                 static_cast<float *>(data_b.value), rows_c.value, cols_c.value,
                 data_c.value, nnz_c, stream.value);
          if (output) {
            auto rows = std::make_unique<int[]>(size_t(m) + 1);
            auto columns = std::make_unique<int[]>(size_t(nnz_c));
            auto values = std::make_unique<float[]>(size_t(nnz_c));
            CHECK_CUDA(cudaMemcpy(rows.get(), rows_c.value,
                                  (size_t(m) + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (nnz_c) {
              CHECK_CUDA(cudaMemcpy(columns.get(), cols_c.value,
                                    size_t(nnz_c) * sizeof(int),
                                    cudaMemcpyDeviceToHost));
              CHECK_CUDA(cudaMemcpy(values.get(), data_c.value,
                                    size_t(nnz_c) * sizeof(float),
                                    cudaMemcpyDeviceToHost));
            }
            *output = {nnz_c, rows.release(), columns.release(), values.release()};
          }
        },
        stream.value, samples ? &benchmark : nullptr);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
