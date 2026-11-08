"""CSF MTTKRP with a dense outer level: D[i,r] = sum_jk A[i,j,k] * B[j,r] * C[k,r]."""

from tvm.script import tir as T


def mttkrp(m, n, p, rank, nnz_j, nnz_k):
    @T.prim_func
    def main(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        d: T.handle,
        indptr_j: T.handle,
        indices_j: T.handle,
        indptr_k: T.handle,
        indices_k: T.handle,
    ):
        T.func_attr(
            {"global_symbol": "mttkrp", "tir.noalias": True, "sparse_tir_level": 2}
        )
        I = T.dense_fixed(m)
        J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
        K = T.sparse_variable(J, (p, nnz_k), (indptr_k, indices_k), "int32")
        N = T.dense_fixed(n)
        P = T.dense_fixed(p)
        R = T.dense_fixed(rank)
        A = T.match_sparse_buffer(a, (I, J, K), "float32")
        B = T.match_sparse_buffer(b, (N, R), "float32")
        C = T.match_sparse_buffer(c, (P, R), "float32")
        D = T.match_sparse_buffer(d, (I, R), "float32")
        with T.sp_iter([I, R, J, K], "SSRR", "mttkrp") as [i, r, j, k]:
            with T.init():
                D[i, r] = 0.0
            D[i, r] = D[i, r] + A[i, j, k] * B[j, r] * C[k, r]

    return main
