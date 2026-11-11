"""Generate SpMV reference archives with the cuSPARSE benchmark kernel."""

from benchmarks.references import generate_references
from systems.cusparse.run import run_spmv
from . import config


def main():
  generate_references("spmv", config.CASES, run_spmv)


if __name__ == "__main__":
  main()
