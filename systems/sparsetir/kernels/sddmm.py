"""Weighted CSR SDDMM: D[i,j] = A[i,j] * sum_k B[i,k] * C[j,k]."""

from tvm.script import tir as T


def sddmm(m, n, rank, nnz):
  @T.prim_func
  def main(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    d: T.handle,
    indptr: T.handle,
    indices: T.handle,
  ):
    T.func_attr({"global_symbol": "sddmm", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    N = T.dense_fixed(n)
    K = T.dense_fixed(rank)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I, K), "float32")
    C = T.match_sparse_buffer(c, (N, K), "float32")
    D = T.match_sparse_buffer(d, (I, J), "float32")
    with T.sp_iter([I, J, K], "SSR", "sddmm") as [i, j, k]:
      with T.init():
        D[i, j] = 0.0
      D[i, j] = D[i, j] + A[i, j] * B[i, k] * C[j, k]

  return main
