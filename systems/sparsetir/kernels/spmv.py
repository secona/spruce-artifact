"""CSR SpMV: y[i] = sum_j A[i,j] * x[j]."""

from tvm.script import tir as T


def spmv(m, n, nnz):
    @T.prim_func
    def main(a: T.handle, x: T.handle, y: T.handle,
             indptr: T.handle, indices: T.handle):
        T.func_attr({"global_symbol": "spmv", "tir.noalias": True, "sparse_tir_level": 2})
        I = T.dense_fixed(m)
        J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
        N = T.dense_fixed(n)
        A = T.match_sparse_buffer(a, (I, J), "float32")
        X = T.match_sparse_buffer(x, (N,), "float32")
        Y = T.match_sparse_buffer(y, (I,), "float32")
        with T.sp_iter([I, J], "SR", "spmv") as [i, j]:
            with T.init():
                Y[i] = 0.0
            Y[i] = Y[i] + A[i, j] * X[j]

    return main
