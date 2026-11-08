// Bridge canonical host CSR arrays to MLIR's sparse tensor runtime and C
// interface.
#include "mlir/ExecutionEngine/SparseTensorRuntime.h"
#include <algorithm>
#include <cassert>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/cuda_utils.h"
#include "../common/execution.h"

using namespace mlir::sparse_tensor;
using MemRef = StridedMemRefType<float, 1>;

extern "C" void _mlir_ciface_spmv(MemRef *, void *, MemRef *, MemRef *);
extern "C" void mgpuSetDefaultDevice(int32_t);

template <typename T> StridedMemRefType<T, 1> view(T *data, int64_t size) {
  return {data, data, 0, {size}, {1}};
}

extern "C" const char *spruce_benchmark_spmv(int device, int m, int n,
                                             int64_t nnz, uint64_t *offsets,
                                             uint64_t *columns, float *values,
                                             float *x, int warmup, int repeat,
                                             double *samples, float *output) {
  static thread_local std::string error;
  try {
    assert(warmup >= 0);
    assert(repeat >= 1);
    assert(samples);
    CHECK_CUDA(cudaSetDevice(device));
    mgpuSetDefaultDevice(device);
    uint64_t sizes[] = {uint64_t(m), uint64_t(n)}, mapping[] = {0, 1};
    LevelType levels[] = {LevelFormat::Dense, LevelFormat::Compressed};
    auto dims = view(sizes, 2);
    auto maps = view(mapping, 2);
    auto types = view(levels, 2);
    intptr_t buffers[] = {reinterpret_cast<intptr_t>(offsets),
                          reinterpret_cast<intptr_t>(columns),
                          reinterpret_cast<intptr_t>(values)};
    std::unique_ptr<void, decltype(&delSparseTensor)> a(
        _mlir_ciface_newSparseTensor(&dims, &dims, &types, &maps, &maps,
                                     OverheadType::kU64, OverheadType::kU64,
                                     PrimaryType::kF32, Action::kPack, buffers),
        delSparseTensor);
    auto input = view(x, n);
    auto y = view(output, m);
    std::vector<float> timings(repeat);
    Benchmark benchmark{warmup, repeat, timings.data()};
    execute(
        [&] {
          MemRef result{};
          _mlir_ciface_spmv(&result, a.get(), &input, &y);
          CHECK_CUDA(cudaDeviceSynchronize());
        },
        0, &benchmark,
        [&] {
          // The MLIR operation accumulates into y; reset it outside timing.
          std::fill(output, output + m, 0.0f);
        },
        [](cudaEvent_t start) {
          // Generated code creates its own streams. Complete the start event
          // before entering it; launch waits for all work before the stop event.
          CHECK_CUDA(cudaEventSynchronize(start));
        });
    std::copy(timings.begin(), timings.end(), samples);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
