#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    assert(status == cudaSuccess);                                            \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    assert(status == CUSPARSE_STATUS_SUCCESS);                                 \
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
