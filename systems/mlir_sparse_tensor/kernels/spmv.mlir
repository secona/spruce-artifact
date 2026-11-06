#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// y[i] += sum_j A[i,j] * x[j]. The caller initializes y to zero.
func.func @spmv(%A: tensor<?x?xf32, #CSR>,
                %x: tensor<?xf32>, %y: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,j) -> (i,j)>,
      affine_map<(i,j) -> (j)>,
      affine_map<(i,j) -> (i)>
    ],
    iterator_types = [
      "parallel",
      "reduction"
    ]
  } ins(%A, %x: tensor<?x?xf32, #CSR>, tensor<?xf32>)
    outs(%y: tensor<?xf32>) {
    ^bb0(%a: f32, %x_value: f32, %y_value: f32):
      %product = arith.mulf %a, %x_value : f32
      %sum = arith.addf %y_value, %product : f32
      linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
