.PHONY: all generated clean
.DELETE_ON_ERROR:

UV ?= uv
PYTHON ?= $(UV) run --locked
SPARSETIR_DIR ?= ../third_party/sparsetir
ARGS ?=

export PYTHONPATH := $(abspath $(SPARSETIR_DIR)/python)$(if $(PYTHONPATH),:$(PYTHONPATH))
export TVM_LIBRARY_PATH := $(abspath $(SPARSETIR_DIR)/build)

all: generated

generated:
	$(PYTHON) experiments.py $(ARGS)

clean:
	rm -rf generated __pycache__
