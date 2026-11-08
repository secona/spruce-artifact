#include "Kernel.h"
#include "backend/output.h"

#include <filesystem>

using namespace nacho;

int main(int argc, char **argv) {
  if (!nacho::parse_command_line(argc, argv))
    return 1;
  // Keep the compiler executable alongside generated kernels.
  std::filesystem::create_directories(nacho::backend::output_directory());

  Format csr_ij = Format::ordered({
      {"i", LevelFormat::Dense},
      {"j", LevelFormat::Compressed_unique},
  }).named("CSR");
  Format csr_jk = Format::ordered({
      {"j", LevelFormat::Dense},
      {"k", LevelFormat::Compressed_unique},
  }).named("CSR");
  TensorType csr_ij_f32 = TensorType(csr_ij, dType::Float32);
  TensorType csr_jk_f32 = TensorType(csr_jk, dType::Float32);
  Expr a_csr_ij = Tensor::make(csr_ij_f32, "a");
  Expr b_csr_jk = Tensor::make(csr_jk_f32, "b");

  Kernel("spgemm")
      .expr(Sum::make("j", a_csr_ij * b_csr_jk))
      .targets({Target::GPU})
      .emit();

  return 0;
}
