#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// D[i,j] = A[i,j] * sum_k B[i,k] * C[j,k], sampled by CSR A.
func.func @sddmm(%A: tensor<?x?xf32, #CSR>,
                 %B: tensor<?x?xf32>,
                 %C: tensor<?x?xf32>) -> tensor<?x?xf32, #CSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %A, %c0 : tensor<?x?xf32, #CSR>
  %n = tensor.dim %A, %c1 : tensor<?x?xf32, #CSR>
  %D = tensor.empty(%m, %n) : tensor<?x?xf32, #CSR>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,j,k) -> (i,j)>,
      affine_map<(i,j,k) -> (i,k)>,
      affine_map<(i,j,k) -> (j,k)>,
      affine_map<(i,j,k) -> (i,j)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "reduction"
    ]
  } ins(%A, %B, %C: tensor<?x?xf32, #CSR>, tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%D: tensor<?x?xf32, #CSR>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %bc = arith.mulf %b, %c : f32
      %abc = arith.mulf %a, %bc : f32
      %sum = arith.addf %d, %abc : f32
      linalg.yield %sum : f32
  } -> tensor<?x?xf32, #CSR>
  return %result : tensor<?x?xf32, #CSR>
}
