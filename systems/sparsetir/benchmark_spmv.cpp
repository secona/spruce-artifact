// C ABI for the SparseTIR SpMV benchmark. Transfers are outside timing.
#include "kernels/common.h"

extern "C" const char *spruce_benchmark_spmv(
    int device, int m, int n, int64_t nnz, const int *offsets,
    const int *columns, const float *values, const float *x, int warmup,
    int repeat, float *samples, float *output) {
  static thread_local std::string error;
  try {
    CHECK_CUDA(cudaSetDevice(device));
    check_tvm(TVMSetStream(kDLCUDA, device, nullptr));
    DeviceArray<int> rows(device, int64_t(m) + 1, offsets);
    DeviceArray<int> cols(device, nnz, columns);
    DeviceArray<float> data(device, nnz, values);
    DeviceArray<float> input(device, n, x);
    DeviceArray<float> result(device, m);
    Kernel kernel("spmv",
                  {&data.tensor, &input.tensor, &result.tensor, &rows.tensor,
                   &cols.tensor},
                  {m, n, int(nnz)});
    benchmark_kernel(kernel, result, warmup, repeat, samples);
    result.copy_to(output);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
