from pathlib import Path

import numpy as np
from scipy.io import mminfo


DATASET_DIR = Path("dataset")
SEED = 1234
RANK = 32


def save_input(path, array):
  if path.exists():
    existing = np.load(path, allow_pickle=False)
    assert existing.shape == array.shape
    assert existing.dtype == np.float32
    assert np.all(np.isfinite(existing))
    print(f"Already generated: {path.name}")
    return
  np.save(path, array, allow_pickle=False)
  print(f"Generated: {path.name}")


def tensor_shape(path):
  """Infer dimensions from one-based coordinates without loading the tensor."""
  shape = [0, 0, 0]
  with path.open() as tensor:
    for line in tensor:
      fields = line.split()
      if not fields:
        continue
      if fields[0].startswith(("#", "%")):
        continue
      assert len(fields) == 4
      indices = [int(value) for value in fields[:3]]
      assert min(indices) >= 1
      shape = [max(size, index) for size, index in zip(shape, indices)]
  assert min(shape) >= 1
  return shape


def generate_inputs(path):
  if path.suffix == ".mtx":
    rows, cols = mminfo(path)[:2]
    x = np.random.default_rng(SEED).random(cols, dtype=np.float32)
    save_input(path.with_name(f"{path.stem}_x.npy"), x)
  else:
    _, rows, cols = tensor_shape(path)

  rng = np.random.default_rng(SEED)
  for name, size in (("B", rows), ("C", cols)):
    factor = rng.random((size, RANK), dtype=np.float32)
    save_input(path.with_name(f"{path.stem}_{name}.npy"), factor)


def main():
  for pattern in ("*.mtx", "*.tns"):
    for path in sorted(DATASET_DIR.glob(pattern)):
      generate_inputs(path)


if __name__ == "__main__":
  main()
