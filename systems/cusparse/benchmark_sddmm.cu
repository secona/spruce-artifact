// C ABI for the SDDMM benchmark runner. Host arrays are copied before timing.
#include "common.cuh"
#include <string>

using namespace spruce::cusparse;
using namespace spruce::cusparse::detail;

extern "C" const char *spruce_benchmark_sddmm(
    int device, int m, int n, int rank, int64_t nnz,
    const int *offsets, const int *columns, const float *values,
    const float *b, const float *c,
    int warmup, int repetitions, float *samples) {
    static thread_local std::string error;
    try {
        check(cudaSetDevice(device));
        Resource<cudaStream_t, cudaStreamDestroy> stream;
        check(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking));
        Buffer rows((size_t(m) + 1) * sizeof(int));
        Buffer cols(size_t(nnz) * sizeof(int)), data(size_t(nnz) * sizeof(float));
        Buffer left(size_t(m) * rank * sizeof(float));
        Buffer right(size_t(n) * rank * sizeof(float));
        Buffer output(size_t(nnz) * sizeof(float));
        check(cudaMemcpy(rows.value, offsets, (size_t(m) + 1) * sizeof(int), cudaMemcpyHostToDevice));
        if (nnz) {
            check(cudaMemcpy(cols.value, columns, size_t(nnz) * sizeof(int), cudaMemcpyHostToDevice));
            check(cudaMemcpy(data.value, values, size_t(nnz) * sizeof(float), cudaMemcpyHostToDevice));
        }
        check(cudaMemcpy(left.value, b, size_t(m) * rank * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(right.value, c, size_t(n) * rank * sizeof(float), cudaMemcpyHostToDevice));
        Benchmark benchmark{warmup, repetitions, samples};
        sddmm(m, n, rank, nnz, static_cast<int *>(rows.value), static_cast<int *>(cols.value),
              static_cast<float *>(data.value), static_cast<float *>(left.value),
              static_cast<float *>(right.value), static_cast<float *>(output.value),
              stream.value, &benchmark);
        return nullptr;
    } catch (const std::exception &exception) {
        error = exception.what();
        return error.c_str();
    }
}
