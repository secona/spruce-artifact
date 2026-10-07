.PHONY: help submodules preflight

help:
	@echo "Available commands:"
	@echo "  make preflight  - Run preflight checks for CLIs"
	@echo "  make submodules - Update submodules"

preflight:
	@git --version || { exit 1; }
	@echo ">>> Good to go!"

submodules:
	git submodule update --init --recursive --depth=1
