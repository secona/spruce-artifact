#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t cuda_check_status = (call);                              \
    if (cuda_check_status != cudaSuccess) {                                    \
      std::fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(cuda_check_status));                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
