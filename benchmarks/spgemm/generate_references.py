"""Generate SpGEMM reference archives with the cuSPARSE benchmark kernel."""

from benchmarks.references import generate_references
from systems.cusparse.run import run_spgemm
from . import config


def main():
  generate_references("spgemm", config.CASES, run_spgemm)


if __name__ == "__main__":
  main()
