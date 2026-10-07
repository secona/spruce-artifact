#include "generated/spgemm_gpu.h"

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main() {
  using Index = spgemm::index_t;
  using Value = spgemm::value_t;

  const Index host_offsets[] = {0, 1};
  const Index host_indices[] = {0};
  const Value host_values[] = {1.0f};
  Index *offsets = nullptr, *indices = nullptr;
  Value *values = nullptr;
  CUDA_CHECK(cudaMalloc(&offsets, sizeof(host_offsets)));
  CUDA_CHECK(cudaMalloc(&indices, sizeof(host_indices)));
  CUDA_CHECK(cudaMalloc(&values, sizeof(host_values)));
  CUDA_CHECK(cudaMemcpy(offsets, host_offsets, sizeof(host_offsets),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(indices, host_indices, sizeof(host_indices),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(values, host_values, sizeof(host_values),
                        cudaMemcpyHostToDevice));

  const spgemm::a_tensor_format<Index, Value> a{1,       1,      offsets, 1,
                                                indices, values, 1};
  const spgemm::b_tensor_format<Index, Value> b{1,       1,      offsets, 1,
                                                indices, values, 1};

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int iterations = 10;
  constexpr int blocks = 256, threads = 256;
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    auto result = spgemm::spgemm_gpu_i32_f32(a, b, blocks, threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf("Execution %d: %.4f ms\n", i + 1, elapsed_ms);

    CUDA_CHECK(cudaFreeAsync(result.dim_k_offsets, 0));
    CUDA_CHECK(cudaFreeAsync(result.dim_k_indices, 0));
    CUDA_CHECK(cudaFreeAsync(result.values, 0));
  }

  CUDA_CHECK(cudaStreamSynchronize(0));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(offsets));
  CUDA_CHECK(cudaFree(indices));
  CUDA_CHECK(cudaFree(values));
  return 0;
}
