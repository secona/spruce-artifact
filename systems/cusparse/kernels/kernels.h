#pragma once

#include <cuda_runtime.h>

// y[i] = sum_j A[i,j] * x[j]. Caller allocates y (m elements).
void spmv(int m, int n, int64_t nnz, int *offsets, int *columns, float *values,
          float *x, float *y, cudaStream_t stream = nullptr);

// d[p] = a[p] * dot(B[i,:], C[j,:]) for CSR entry p at (i,j).
// Caller allocates d (nnz elements); output uses the input CSR structure.
void sddmm(int m, int n, int rank, int64_t nnz, int *offsets, int *columns,
           float *a, float *b, float *c, float *d,
           cudaStream_t stream = nullptr);

// C = A (m x k) * B (k x n). Output references are host variables.
// On success, caller owns the three output arrays and frees them with cudaFree.
void spgemm(int m, int k, int n, int64_t nnz_a, int *offsets_a, int *columns_a,
            float *values_a, int64_t nnz_b, int *offsets_b, int *columns_b,
            float *values_b, int *&offsets_c, int *&columns_c, float *&values_c,
            int64_t &nnz_c, cudaStream_t stream = nullptr);

// D[i,r] = sum_jk A[i,j,k] * B[j,r] * C[k,r]. A is mode-0 CSR with
// compact columns mapped to (pair_j, pair_k). Caller allocates d (m*rank).
void mttkrp(int m, int rank, int64_t nnz, int pairs, int *offsets,
            int *columns, float *values, int *pair_j, int *pair_k, float *b,
            float *c, float *d, cudaStream_t stream = nullptr);
