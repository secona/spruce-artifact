.PHONY: all generated clean
.DELETE_ON_ERROR:

MLIR_OPT ?= ../../llvm-project/build/bin/mlir-opt

GPU_PIPELINE := builtin.module(pre-sparsification-rewrite,sparse-reinterpret-map,\
                sparsification{parallelization-strategy=dense-any-loop},\
                sparse-gpu-codegen)

all: generated/spmv.mlir generated/mttkrp.mlir

generated: all

generated/:
	mkdir -p $@

generated/%.mlir: %.mlir Makefile | generated/
	$(MLIR_OPT) $< --pass-pipeline="$(GPU_PIPELINE)" -o $@

clean:
	rm -rf generated
