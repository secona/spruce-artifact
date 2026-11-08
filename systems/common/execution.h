#include <cuda_runtime.h>

struct Benchmark {
  int warmup;
  int repetitions;
  float *samples;
};

// Prepare runs outside timing. BeforeLaunch can order work on other streams
// after the start event; launch must complete that work before returning.
template <typename Function, typename Prepare, typename BeforeLaunch>
void execute(Function launch, cudaStream_t stream, const Benchmark *benchmark,
             Prepare prepare, BeforeLaunch before_launch) {
  if (!benchmark) {
    prepare();
    launch();
    return;
  }

  if (benchmark->warmup < 0 || benchmark->repetitions < 1 ||
      !benchmark->samples)
    throw std::invalid_argument("Invalid benchmark configuration");

  CudaEvent start, stop;

  CHECK_CUDA(cudaEventCreate(&start.value));
  CHECK_CUDA(cudaEventCreate(&stop.value));

  for (int i = 0; i < benchmark->warmup; ++i) {
    prepare();
    launch();
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (int i = 0; i < benchmark->repetitions; ++i) {
    prepare();
    CHECK_CUDA(cudaEventRecord(start.value, stream));
    before_launch(start.value);
    launch();
    CHECK_CUDA(cudaEventRecord(stop.value, stream));

    CHECK_CUDA(cudaEventSynchronize(stop.value));
    CHECK_CUDA(
        cudaEventElapsedTime(&benchmark->samples[i], start.value, stop.value));
  }
}

template <typename Function>
void execute(Function launch, cudaStream_t stream, const Benchmark *benchmark) {
  execute(launch, stream, benchmark, [] {}, [](cudaEvent_t) {});
}
