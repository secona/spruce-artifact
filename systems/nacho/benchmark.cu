// The same generated entry point used by Nacho's gpu_spgemm_f32 binding.
#include "spgemm_gpu.h"
#include <limits>
#include <stdexcept>
#include <string>

#include "../common/cuda_utils.h"

struct HostCsr {
  int rows, cols;
  int64_t nnz;
  const int *offsets, *columns;
  const float *values;
};

namespace {
template <typename T, auto destroy> struct Resource {
  T value = nullptr;
  Resource() = default;
  Resource(const Resource &) = delete;
  Resource &operator=(const Resource &) = delete;
  ~Resource() {
    if (value)
      destroy(value);
  }
};

struct Buffer : Resource<void *, cudaFree> {
  explicit Buffer(size_t bytes) {
    CHECK_CUDA(cudaMalloc(&value, bytes ? bytes : 1));
  }
};

struct DeviceCsr {
  Buffer offsets, columns, values;
  explicit DeviceCsr(const HostCsr &a)
      : offsets((size_t(a.rows) + 1) * sizeof(int)),
        columns(size_t(a.nnz) * sizeof(int)),
        values(size_t(a.nnz) * sizeof(float)) {
    CHECK_CUDA(cudaMemcpy(offsets.value, a.offsets,
                          (size_t(a.rows) + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (a.nnz) {
      CHECK_CUDA(cudaMemcpy(columns.value, a.columns,
                            size_t(a.nnz) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(values.value, a.values,
                            size_t(a.nnz) * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
  }
};

struct Product {
  spgemm::Z_tensor_format<int32_t, float> value{};
  ~Product() {
    // Match the generated kernel's stream. Output release is outside timing.
    if (value.dim_k_offsets)
      cudaFreeAsync(value.dim_k_offsets, 0);
    if (value.dim_k_indices)
      cudaFreeAsync(value.dim_k_indices, 0);
    if (value.values)
      cudaFreeAsync(value.values, 0);
  }
};
} // namespace

extern "C" const char *spruce_benchmark_spgemm(int device, const HostCsr *a,
                                               const HostCsr *b, int blocks,
                                               int threads, int warmup,
                                               int repeat, float *samples) {
  static thread_local std::string error;
  try {
    if (!a || !b || !samples || a->rows < 1 || a->cols < 1 || b->cols < 1 ||
        a->cols != b->rows || a->nnz < 0 || b->nnz < 0 ||
        a->nnz > std::numeric_limits<int32_t>::max() ||
        b->nnz > std::numeric_limits<int32_t>::max() || warmup < 0 ||
        repeat < 1 || blocks < 1 || threads < 1 ||
        int64_t(blocks) * threads >= std::numeric_limits<int32_t>::max())
      throw std::invalid_argument(
          "Invalid SpGEMM inputs or benchmark configuration");
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties;
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (threads > properties.maxThreadsPerBlock ||
        blocks > properties.maxGridSize[0])
      throw std::invalid_argument("Launch geometry exceeds device limits");
    DeviceCsr da(*a), db(*b);
    spgemm::a_tensor_format<int32_t, float> input_a{
        a->rows,
        a->cols,
        static_cast<int *>(da.offsets.value),
        static_cast<int32_t>(a->nnz),
        static_cast<int *>(da.columns.value),
        static_cast<float *>(da.values.value),
        static_cast<int32_t>(a->nnz)};
    spgemm::b_tensor_format<int32_t, float> input_b{
        b->rows,
        b->cols,
        static_cast<int *>(db.offsets.value),
        static_cast<int32_t>(b->nnz),
        static_cast<int *>(db.columns.value),
        static_cast<float *>(db.values.value),
        static_cast<int32_t>(b->nnz)};
    Resource<cudaEvent_t, cudaEventDestroy> start, stop;
    CHECK_CUDA(cudaEventCreate(&start.value));
    CHECK_CUDA(cudaEventCreate(&stop.value));
    for (int i = 0; i < warmup; ++i) {
      Product result;
      result.value =
          spgemm::spgemm_gpu_i32_f32(input_a, input_b, blocks, threads);
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < repeat; ++i) {
      // Complete the preceding result's release before the start event.
      CHECK_CUDA(cudaDeviceSynchronize());
      Product result;
      CHECK_CUDA(cudaEventRecord(start.value, 0));
      result.value =
          spgemm::spgemm_gpu_i32_f32(input_a, input_b, blocks, threads);
      CHECK_CUDA(cudaEventRecord(stop.value, 0));
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventSynchronize(stop.value));
      CHECK_CUDA(cudaEventElapsedTime(&samples[i], start.value, stop.value));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return nullptr;
  } catch (const std::exception &exception) {
    error = exception.what();
    return error.c_str();
  }
}
