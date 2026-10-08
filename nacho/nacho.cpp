#include "Kernel.h"
#include "backend/output.h"

#include <iostream>

using namespace nacho;

int main(int argc, char **argv) {
  if (!nacho::parse_command_line(argc, argv))
    return 1;
  nacho::backend::reset_output_directory();

  Format csr_ij = Format::ordered({
      {"i", LevelFormat::Dense},
      {"j", LevelFormat::Compressed_unique},
  }).named("CSR");
  Format csr_jk = Format::ordered({
      {"j", LevelFormat::Dense},
      {"k", LevelFormat::Compressed_unique},
  }).named("CSR");
  Format dense_j = Format::ordered({
      {"j", LevelFormat::Dense},
  }).named("DenseVector");
  Format dense_ik = Format::ordered({
      {"i", LevelFormat::Dense},
      {"k", LevelFormat::Dense},
  }).named("DenseMatrix");
  Format dense_jk = Format::ordered({
      {"j", LevelFormat::Dense},
      {"k", LevelFormat::Dense},
  }).named("DenseMatrix");
  Format csf_ijk = Format::ordered({
      {"i", LevelFormat::Compressed_unique},
      {"j", LevelFormat::Compressed_unique},
      {"k", LevelFormat::Compressed_unique},
  }).named("CSF3");
  Format dense_jr = Format::ordered({
      {"j", LevelFormat::Dense},
      {"r", LevelFormat::Dense},
  }).named("DenseMatrix");
  Format dense_kr = Format::ordered({
      {"k", LevelFormat::Dense},
      {"r", LevelFormat::Dense},
  }).named("DenseMatrix");

  TensorType csr_ij_f32 = TensorType(csr_ij, dType::Float32);
  TensorType csr_jk_f32 = TensorType(csr_jk, dType::Float32);
  TensorType dense_j_f32 = TensorType(dense_j, dType::Float32);
  TensorType dense_ik_f32 = TensorType(dense_ik, dType::Float32);
  TensorType dense_jk_f32 = TensorType(dense_jk, dType::Float32);
  TensorType csf_ijk_f32 = TensorType(csf_ijk, dType::Float32);
  TensorType dense_jr_f32 = TensorType(dense_jr, dType::Float32);
  TensorType dense_kr_f32 = TensorType(dense_kr, dType::Float32);

  Expr a_csr_ij = Tensor::make(csr_ij_f32, "a");
  Expr b_csr_jk = Tensor::make(csr_jk_f32, "b");
  Expr b_dense_j = Tensor::make(dense_j_f32, "b");
  Expr b_dense_ik = Tensor::make(dense_ik_f32, "b");
  Expr c_dense_jk = Tensor::make(dense_jk_f32, "c");
  Expr a_csf_ijk = Tensor::make(csf_ijk_f32, "a");
  Expr b_dense_jr = Tensor::make(dense_jr_f32, "b");
  Expr c_dense_kr = Tensor::make(dense_kr_f32, "c");

  Kernel("spgemm")
      .expr(Sum::make("j", a_csr_ij * b_csr_jk))
      .targets({Target::GPU})
      .emit();

  // SpMV: Z[i] = sum_j A[i,j] * b[j].
  // Disabled: code generation asserts that reduction index j is missing.

  // Kernel("spmv")
  //     .expr(Sum::make("j", a_csr_ij * b_dense_j))
  //     .targets({Target::GPU})
  //     .emit();

  // SDDMM: Z[i,j] = A[i,j] * sum_k B[i,k] * C[j,k].
  // Disabled: code generation asserts that reduction index k is missing.

  // Kernel("sddmm")
  //     .expr(Sum::make("k", a_csr_ij * b_dense_ik * c_dense_jk))
  //     .targets({Target::GPU})
  //     .emit();

  // MTTKRP: Z[i,r] = sum_j sum_k A[i,j,k] * B[j,r] * C[k,r].
  // Disabled: code generation asserts that the j loop simplifies to empty.

  // Kernel("mttkrp")
  //     .expr(Sum::make("j", Sum::make("k", a_csf_ijk * b_dense_jr * c_dense_kr)))
  //     .targets({Target::GPU})
  //     .emit();

  return 0;
}
