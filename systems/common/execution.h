#include <cuda_runtime.h>

struct Benchmark {
  int warmup;
  int repetitions;
  float *samples;
};

template <typename Function>
void execute(Function launch, cudaStream_t stream, const Benchmark *benchmark) {
  if (!benchmark) {
    launch();
    return;
  }

  if (benchmark->warmup < 0 || benchmark->repetitions < 1 ||
      !benchmark->samples)
    throw std::invalid_argument("Invalid benchmark configuration");

  CudaEvent start, stop;

  CHECK_CUDA(cudaEventCreate(&start.value));
  CHECK_CUDA(cudaEventCreate(&stop.value));

  for (int i = 0; i < benchmark->warmup; ++i)
    launch();

  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (int i = 0; i < benchmark->repetitions; ++i) {
    CHECK_CUDA(cudaEventRecord(start.value, stream));
    launch();
    CHECK_CUDA(cudaEventRecord(stop.value, stream));

    CHECK_CUDA(cudaEventSynchronize(stop.value));
    CHECK_CUDA(
        cudaEventElapsedTime(&benchmark->samples[i], start.value, stop.value));
  }
}
