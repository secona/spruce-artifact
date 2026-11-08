// Bridge host CSF arrays with a dense outer level to MLIR's C interface.
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
using MemRef = StridedMemRefType<float, 2>;

extern "C" void _mlir_ciface_mttkrp(MemRef *, void *, MemRef *, MemRef *,
                                    MemRef *);
extern "C" void mgpuSetDefaultDevice(int32_t);

template <typename T> StridedMemRefType<T, 1> view(T *data, int64_t size) {
  return {data, data, 0, {size}, {1}};
}

MemRef matrix_view(float *data, int64_t rows, int64_t columns) {
  return {data, data, 0, {rows, columns}, {columns, 1}};
}

extern "C" const char *spruce_benchmark_mttkrp(
    int device, int m, int n, int p, int rank,
    uint64_t *positions_j, uint64_t *coordinates_j,
    uint64_t *positions_k, uint64_t *coordinates_k, float *values, float *b,
    float *c, int warmup, int repeat, double *samples, float *output) {
  static thread_local std::string error;
  try {
    assert(warmup >= 0);
    assert(repeat >= 1);
    assert(samples);
    CHECK_CUDA(cudaSetDevice(device));
    mgpuSetDefaultDevice(device);
    uint64_t sizes[] = {uint64_t(m), uint64_t(n), uint64_t(p)};
    uint64_t mapping[] = {0, 1, 2};
    LevelType levels[] = {LevelFormat::Dense, LevelFormat::Compressed,
                          LevelFormat::Compressed};
    auto dims = view(sizes, 3);
    auto maps = view(mapping, 3);
    auto types = view(levels, 3);
    intptr_t buffers[] = {reinterpret_cast<intptr_t>(positions_j),
                          reinterpret_cast<intptr_t>(coordinates_j),
                          reinterpret_cast<intptr_t>(positions_k),
                          reinterpret_cast<intptr_t>(coordinates_k),
                          reinterpret_cast<intptr_t>(values)};
    std::unique_ptr<void, decltype(&delSparseTensor)> a(
        _mlir_ciface_newSparseTensor(&dims, &dims, &types, &maps, &maps,
                                     OverheadType::kU64, OverheadType::kU64,
                                     PrimaryType::kF32, Action::kPack, buffers),
        delSparseTensor);
    auto input_b = matrix_view(b, n, rank);
    auto input_c = matrix_view(c, p, rank);
    auto d = matrix_view(output, m, rank);
    std::vector<float> timings(repeat);
    Benchmark benchmark{warmup, repeat, timings.data()};
    execute(
        [&] {
          MemRef result{};
          _mlir_ciface_mttkrp(&result, a.get(), &input_b, &input_c, &d);
          CHECK_CUDA(cudaDeviceSynchronize());
        },
        0, &benchmark,
        [&] {
          // The MLIR operation accumulates into D; reset it outside timing.
          std::fill(output, output + int64_t(m) * rank, 0.0f);
        },
        [](cudaEvent_t start) {
          CHECK_CUDA(cudaEventSynchronize(start));
        });
    std::copy(timings.begin(), timings.end(), samples);
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
