"""Generate SDDMM reference archives with the cuSPARSE benchmark kernel."""

from benchmarks.references import generate_references
from systems.cusparse.run import run_sddmm
from . import config


def main():
  generate_references("sddmm", config.CASES, run_sddmm)


if __name__ == "__main__":
  main()
