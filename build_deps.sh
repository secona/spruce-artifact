set -euo pipefail

nacho() {
	echo "--- Building Nacho"
	cmake -S third_party/nacho -B third_party/nacho/build
	cmake --build third_party/nacho/build
}

sparsetir() {
	echo "--- Building SparseTIR"
	cmake -S third_party/sparsetir -B third_party/sparsetir/build \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
		-DLLVM_DIR="${LLVM_14_BUILD_DIR}/lib/cmake/llvm" \
		-DUSE_LLVM="${LLVM_14_BUILD_DIR}/bin/llvm-config --link-shared" \
		-DUSE_CUDA=ON
	cmake --build third_party/sparsetir/build --parallel "$(nproc)"
}

case "${1:-help}" in
	nacho)
		nacho
		;;

	sparsetir)
		sparsetir
		;;

	all)
		nacho
		sparsetir
		;;

	help)
		echo "./build_deps.sh [option]"
		echo "available: nacho, sparsetir, all"
esac
