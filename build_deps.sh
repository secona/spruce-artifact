set -euo pipefail

nacho() {
	echo "--- Building Nacho"
	cmake -S third_party/nacho -B third_party/nacho/build
	cmake --build third_party/nacho/build
}

sparsetir() {
	echo "--- Building SparseTIR"
	cmake -S third_party/sparsetir -B third_party/sparsetir/build -DUSE_CUDA=ON -DUSE_LLVM=ON
	cmake --build third_party/sparsetir/build
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
