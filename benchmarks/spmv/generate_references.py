import argparse
from pathlib import Path

import numpy as np
from scipy import io, sparse
import torch


ROOT = Path(__file__).resolve().parents[2]
CURR = Path(__file__).resolve().parent


def get_matrices():
    matrices_txt = CURR / "matrices.txt"
    return matrices_txt.read_text().splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    assert args.seed >= 0
    assert torch.cuda.is_available()
    assert 0 <= args.device < torch.cuda.device_count()

    for name in get_matrices():
        matrixfile = ROOT / "dataset" / name
        run(matrixfile, args.seed, args.device)


def run(matrixfile, seed, device):
    matrix = io.mmread(matrixfile, spmatrix=False)
    assert not np.iscomplexobj(matrix)

    a = sparse.csr_matrix(matrix, dtype=np.float32)
    a.sum_duplicates()
    a.sort_indices()

    assert min(a.shape) >= 1 and max(*a.shape, a.nnz) <= 2**31 - 1
    assert np.all(np.isfinite(a.data))

    offsets = np.ascontiguousarray(a.indptr, dtype=np.int32)
    columns = np.ascontiguousarray(a.indices, dtype=np.int32)
    values = np.ascontiguousarray(a.data, dtype=np.float32)

    x = np.random.default_rng(seed).random(a.shape[1], dtype=np.float32)
    device = torch.device("cuda", device)
    a_gpu = torch.sparse_csr_tensor(
        torch.from_numpy(offsets).to(device),
        torch.from_numpy(columns).to(device),
        torch.from_numpy(values).to(device),
        size=a.shape,
        device=device,
        check_invariants=True,
    )
    with torch.inference_mode():
        y = torch.mv(a_gpu, torch.from_numpy(x).to(device)).cpu().numpy()

    output = CURR / "references" / f"{matrixfile.stem}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        shape=np.asarray(a.shape, dtype=np.int64),
        indptr=offsets, indices=columns, data=values, x=x, y=y,
        seed=str(seed), matrix=str(matrixfile.resolve()),
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device),
    )
    print(f"Saved {output}: shape={a.shape}, nnz={a.nnz}, device={device}")


if __name__ == "__main__":
    main()
