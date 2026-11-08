// C ABI for SparseTIR MTTKRP on CSF inputs with a dense outer level.
#include "kernels/common.h"

extern "C" const char *spruce_benchmark_mttkrp(
    int device, int m, int n, int p, int rank, int64_t nnz_j, int64_t nnz_k,
    const int *positions_j, const int *coordinates_j, const int *positions_k,
    const int *coordinates_k, const float *values, const float *b,
    const float *c, int warmup, int repeat, float *samples, float *output) {
  static thread_local std::string error;
  try {
    CHECK_CUDA(cudaSetDevice(device));
    check_tvm(TVMSetStream(kDLCUDA, device, nullptr));
    DeviceArray<int> rows_j(device, int64_t(m) + 1, positions_j);
    DeviceArray<int> cols_j(device, nnz_j, coordinates_j);
    DeviceArray<int> rows_k(device, nnz_j + 1, positions_k);
    DeviceArray<int> cols_k(device, nnz_k, coordinates_k);
    DeviceArray<float> data(device, nnz_k, values);
    DeviceArray<float> left(device, int64_t(n) * rank, b);
    DeviceArray<float> right(device, int64_t(p) * rank, c);
    DeviceArray<float> result(device, int64_t(m) * rank);
    Kernel kernel("mttkrp",
                  {&data.tensor, &left.tensor, &right.tensor, &result.tensor,
                   &rows_j.tensor, &cols_j.tensor, &rows_k.tensor, &cols_k.tensor},
                  {m, n, p, rank, int(nnz_j), int(nnz_k)});
    benchmark_kernel(kernel, result, warmup, repeat, samples);
    result.copy_to(output);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
