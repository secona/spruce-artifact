// C ABI for the existing weighted SparseTIR SDDMM kernel.
#include "kernels/common.h"

extern "C" const char *spruce_benchmark_sddmm(
    int device, int m, int n, int rank, int64_t nnz, const int *offsets,
    const int *columns, const float *values, const float *b, const float *c,
    int warmup, int repeat, float *samples, float *output) {
  static thread_local std::string error;
  try {
    CHECK_CUDA(cudaSetDevice(device));
    check_tvm(TVMSetStream(kDLCUDA, device, nullptr));
    DeviceArray<int> rows(device, int64_t(m) + 1, offsets);
    DeviceArray<int> cols(device, nnz, columns);
    DeviceArray<float> data(device, nnz, values);
    DeviceArray<float> left(device, int64_t(m) * rank, b);
    DeviceArray<float> right(device, int64_t(n) * rank, c);
    DeviceArray<float> result(device, nnz);
    Kernel kernel("sddmm",
                  {&data.tensor, &left.tensor, &right.tensor, &result.tensor,
                   &rows.tensor, &cols.tensor},
                  {m, n, rank, int(nnz)});
    benchmark_kernel(kernel, result, warmup, repeat, samples);
    result.copy_to(output);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
