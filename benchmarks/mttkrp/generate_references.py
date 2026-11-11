"""Generate MTTKRP reference archives with the cuSPARSE benchmark kernel."""

from benchmarks.references import generate_references
from systems.cusparse.run import run_mttkrp
from . import config


def main():
  generate_references("mttkrp", config.CASES, run_mttkrp)


if __name__ == "__main__":
  main()
