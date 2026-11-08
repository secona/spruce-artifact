#CSF = #sparse_tensor.encoding<{
  map = (i, j, k) -> (i : dense, j : compressed, k : compressed)
}>

// D[i,r] += sum_j sum_k A[i,j,k] * B[j,r] * C[k,r].
// The caller initializes the dense output D to zero.
func.func @mttkrp(%A: tensor<?x?x?xf32, #CSF>,
                  %B: tensor<?x?xf32>, %C: tensor<?x?xf32>,
                  %D: tensor<?x?xf32>) -> tensor<?x?xf32>
    attributes {llvm.emit_c_interface} {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,r,j,k) -> (i,j,k)>,
      affine_map<(i,r,j,k) -> (j,r)>,
      affine_map<(i,r,j,k) -> (k,r)>,
      affine_map<(i,r,j,k) -> (i,r)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "reduction",
      "reduction"
    ]
  } ins(%A, %B, %C: tensor<?x?x?xf32, #CSF>, tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%D: tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32, %d: f32):
      %ab = arith.mulf %a, %b : f32
      %abc = arith.mulf %ab, %c : f32
      %sum = arith.addf %d, %abc : f32
      linalg.yield %sum : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
