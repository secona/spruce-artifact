.PHONY: help submodules preflight build-nacho

help:
	@echo "Available commands:"
	@echo "  make preflight   - Run preflight checks for CLIs"
	@echo "  make submodules  - Update submodules"
	@echo "  make build-nacho - Build the Nacho compiler"

preflight:
	@cmake --version | head -n 1
	@git --version
	@echo ">>> Good to go!"

submodules:
	git submodule update --init --recursive --depth=1

build-nacho:
	cmake -S third_party/nacho -B third_party/nacho/build
	cmake --build third_party/nacho/build
