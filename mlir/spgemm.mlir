#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// C[i,k] = sum_j A[i,j] * B[j,k], with CSR inputs and output.
func.func @spgemm(%A: tensor<?x?xf32, #CSR>,
                  %B: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32, #CSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %A, %c0 : tensor<?x?xf32, #CSR>
  %n = tensor.dim %B, %c1 : tensor<?x?xf32, #CSR>
  %C = tensor.empty(%m, %n) : tensor<?x?xf32, #CSR>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,k,j) -> (i,j)>,
      affine_map<(i,k,j) -> (j,k)>,
      affine_map<(i,k,j) -> (i,k)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "reduction"
    ]
  } ins(%A, %B: tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR>)
    outs(%C: tensor<?x?xf32, #CSR>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<?x?xf32, #CSR>
  return %result : tensor<?x?xf32, #CSR>
}
