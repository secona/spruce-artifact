#pragma once

#include <tvm/runtime/c_runtime_api.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../common/cuda_utils.h"
#include "../../common/execution.h"

inline void check_tvm(int status) {
  if (status != 0)
    throw std::runtime_error(TVMGetLastError());
}

// The lowered sparse buffers use flat float32/int32 arrays. Own their device
// storage and keep DLTensor metadata alive for TVM's packed calling convention.
template <typename T> struct DeviceArray {
  Buffer buffer;
  int64_t size;
  DLTensor tensor;

  DeviceArray(int device, int64_t size, const T *input = nullptr)
      : buffer(size_t(size) * sizeof(T)), size(size),
        tensor{buffer.value, {kDLCUDA, device}, 1,
               {std::is_same_v<T, float> ? uint8_t(kDLFloat) : uint8_t(kDLInt),
                32, 1},
               &this->size, nullptr, 0} {
    if (input && size)
      CHECK_CUDA(cudaMemcpy(buffer.value, input, size_t(size) * sizeof(T),
                            cudaMemcpyHostToDevice));
  }

  void copy_to(T *output) const {
    if (size)
      CHECK_CUDA(cudaMemcpy(output, buffer.value, size_t(size) * sizeof(T),
                            cudaMemcpyDeviceToHost));
  }
};

struct Kernel {
  Resource<TVMModuleHandle, TVMModFree> module;
  Resource<TVMFunctionHandle, TVMFuncFree> function;
  std::vector<TVMValue> arguments;
  std::vector<int> types;

  Kernel(const char *operation, std::initializer_list<DLTensor *> arrays,
         std::initializer_list<int> dimensions) {
    std::string path = std::string("generated/sparsetir/") + operation + "_kernel.so";
    check_tvm(TVMModLoadFromFile(path.c_str(), "so", &module.value));
    check_tvm(TVMModGetFunction(module.value, operation, 0, &function.value));
    if (!function.value)
      throw std::runtime_error("Missing SparseTIR function: " + std::string(operation));
    for (auto *array : arrays) {
      TVMValue value{};
      value.v_handle = array;
      arguments.push_back(value);
      types.push_back(kTVMDLTensorHandle);
    }
    for (int dimension : dimensions) {
      TVMValue value{};
      value.v_int64 = dimension;
      arguments.push_back(value);
      types.push_back(kTVMArgInt);
    }
  }

  void operator()() {
    TVMValue result{};
    int result_type;
    check_tvm(TVMFuncCall(function.value, arguments.data(), types.data(),
                          int(arguments.size()), &result, &result_type));
    CHECK_CUDA(cudaGetLastError());
  }
};

template <typename T>
void benchmark_kernel(Kernel &kernel, DeviceArray<T> &output, int warmup,
                      int repeat, float *samples) {
  assert(warmup >= 0 && repeat >= 1 && samples);
  auto prepare = [&] {
    // SparseTIR's reduction initialization is inside the sparse loop. Empty
    // rows never enter it, so initialize the output outside each timed call.
    CHECK_CUDA(cudaMemsetAsync(output.buffer.value, 0,
                               size_t(output.size) * sizeof(T), 0));
  };
  prepare();
  kernel(); // Initialize TVM's CUDA module before measuring any execution.
  CHECK_CUDA(cudaStreamSynchronize(0));
  Benchmark benchmark{warmup, repeat, samples};
  execute([&] { kernel(); }, 0, &benchmark, prepare, [](cudaEvent_t) {});
}
