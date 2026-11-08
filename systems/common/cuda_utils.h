#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("cuda error\n");                                                  \
      throw std::runtime_error(cudaGetErrorString(status));                    \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("cusparse error\n");                                              \
      throw std::runtime_error(cusparseGetErrorString(status));                \
    }                                                                          \
  }

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

using CudaEvent = Resource<cudaEvent_t, cudaEventDestroy>;

struct Buffer : Resource<void *, cudaFree> {
  explicit Buffer(size_t bytes) {
    CHECK_CUDA(cudaMalloc(&value, bytes ? bytes : 1));
  }
  template <typename T> T *release() {
    auto result = static_cast<T *>(value);
    value = nullptr;
    return result;
  }
};
