.PHONY: help submodules preflight build-nacho build-sparsetir

SPARSETIR_CMAKE_FLAGS ?= -DUSE_CUDA=ON -DUSE_LLVM=ON

help:
	@echo "Available commands:"
	@echo "  make preflight   - Run preflight checks for CLIs"
	@echo "  make submodules  - Update submodules"
	@echo "  make build-nacho - Build the Nacho compiler"
	@echo "  make build-sparsetir - Build SparseTIR (TVM) with CUDA and LLVM"

preflight:
	@cmake --version | head -n 1
	@git --version
	@echo ">>> Good to go!"

submodules:
	git submodule update --init --recursive --depth=1

build-nacho:
	cmake -S third_party/nacho -B third_party/nacho/build
	cmake --build third_party/nacho/build

build-sparsetir:
	cmake -S third_party/sparsetir -B third_party/sparsetir/build $(SPARSETIR_CMAKE_FLAGS)
	cmake --build third_party/sparsetir/build
