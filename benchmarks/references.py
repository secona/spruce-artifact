"""Generate reference archives using the existing cuSPARSE runners."""

import argparse
from pathlib import Path

import numpy as np


def generate_reference(operation, paths, runner, device, output_dir):
  arrays = runner(*paths, repeat=1, device=device, warmup=0, reference=True)
  output_key = {"spmv": "y", "sddmm": "d", "spgemm": "data", "mttkrp": "d"}[operation]
  if not np.all(np.isfinite(arrays[output_key])):
    raise ValueError(f"cuSPARSE {operation} produced nonfinite reference values")

  output = output_dir / ("__".join(path.stem for path in paths) + ".npz")
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    **arrays,
    backend="cusparse",
    operation=operation,
    device=device,
    operands=np.asarray([str(path.resolve()) for path in paths]),
  )
  print(f"Saved {output}: device=cuda:{device}")
  return output


def generate_references(operation, cases, runner):
  parser = argparse.ArgumentParser(
    description=f"Generate {operation.upper()} reference outputs with cuSPARSE."
  )
  parser.add_argument("--device", type=int, default=0, help="CUDA device index")
  parser.add_argument(
    "--output-dir", type=Path,
    default=Path("benchmarks") / operation / "references",
  )
  args = parser.parse_args()
  if not 0 <= args.device <= 2**31 - 1:
    parser.error("--device must be a nonnegative int32 CUDA device index")

  for filenames in cases:
    paths = [Path("dataset") / filename for filename in filenames]
    generate_reference(operation, paths, runner, args.device, args.output_dir)
