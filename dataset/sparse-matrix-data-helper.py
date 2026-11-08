import argparse
import gzip
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


MATRICES_DIR = Path("dataset")
SEED = 1234
SDDMM_RANK = 32
MTTKRP_RANK = 32

# Add entries with a URL and an output filename ending in .mtx or .tns.
# For tar archives, also specify the exact archive_member to extract.
# Direct files and standalone .gz files do not need archive_member.
MATRICES = [
    {
        "url": "https://www.cise.ufl.edu/research/sparse/MM/HB/bcsstk01.tar.gz",
        "filename": "bcsstk01.mtx",
        "archive_member": "bcsstk01/bcsstk01.mtx",
    },
    {
        "url": "https://www.cise.ufl.edu/research/sparse/MM/HB/bcsstk17.tar.gz",
        "filename": "bcsstk17.mtx",
        "archive_member": "bcsstk17/bcsstk17.mtx",
    },
    {
        "url": "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_4-4-4.tns.gz",
        "filename": "matmul_4-4-4.tns",
    },
    {
        "url": "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz",
        "filename": "nell-2.tns",
    },
]


def matrix_path(matrix):
    """Resolve a catalog entry's output filename inside MATRICES_DIR."""
    filename = matrix["filename"]
    if Path(filename).name != filename or Path(filename).suffix not in (".mtx", ".tns"):
        raise ValueError(f"Expected a .mtx or .tns filename, got {filename!r}")
    return MATRICES_DIR / filename


def is_downloaded(path):
    """Treat missing or empty files as incomplete downloads."""
    return path.is_file() and path.stat().st_size > 0


def download_matrix(matrix):
    destination = matrix_path(matrix)
    if is_downloaded(destination):
        print(f"Already downloaded: {destination.name}")
        return

    MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...", flush=True)
    # Stage the download and extraction so failures cannot leave a final file.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=MATRICES_DIR) as temp_dir:
        source = Path(temp_dir) / "download"
        output = Path(temp_dir) / destination.name
        with urlopen(matrix["url"], timeout=60) as response, source.open("wb") as target:
            shutil.copyfileobj(response, target)

        if "archive_member" in matrix:
            with tarfile.open(source, "r:*") as archive:
                try:
                    member = archive.getmember(matrix["archive_member"])
                except KeyError as error:
                    raise ValueError(f"Archive member missing: {matrix['archive_member']}") from error
                if not member.isfile():
                    raise ValueError(f"Archive member is not a regular file: {member.name}")
                # Copy only this member; never extract archive paths to disk.
                with archive.extractfile(member) as data, output.open("wb") as target:
                    shutil.copyfileobj(data, target)
        elif urlparse(matrix["url"]).path.endswith(".gz"):
            with gzip.open(source, "rb") as data, output.open("wb") as target:
                shutil.copyfileobj(data, target)
        else:
            source.replace(output)

        if not is_downloaded(output):
            raise ValueError("Downloaded file is empty")
        output.replace(destination)
    print(f"Downloaded: {destination}")


def generate_vector(path):
    """Save one float32 SpMV input per matrix and seed, including cached matrices."""
    import numpy as np
    from scipy.io import mminfo

    destination = path.with_name(f"{path.stem}_x.npy")
    columns = mminfo(path)[1]
    if is_downloaded(destination):
        x = np.load(destination, allow_pickle=False)
        if x.shape != (columns,) or x.dtype != np.float32 or not np.all(np.isfinite(x)):
            raise ValueError(f"Invalid SpMV vector: {destination}")
        print(f"Already generated: {destination.name}")
        return

    x = np.random.default_rng(SEED).random(columns, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix=".vector-", dir=MATRICES_DIR) as temp_dir:
        output = Path(temp_dir) / destination.name
        np.save(output, x, allow_pickle=False)
        output.replace(destination)
    print(f"Generated: {destination}")


def generate_factors(path):
    """Save reproducible row-major float32 factors for SDDMM or mode-0 MTTKRP."""
    import numpy as np
    from scipy.io import mminfo

    if path.suffix == ".tns":
        dimensions, _ = tensor_stats(path)
        if len(dimensions) != 3:
            raise ValueError(f"MTTKRP requires a third-order tensor: {path}")
        rows, columns = dimensions[1:]
        rank = MTTKRP_RANK
        operation = "MTTKRP"
    else:
        rows, columns = mminfo(path)[:2]
        rank = SDDMM_RANK
        operation = "SDDMM"
    rng = np.random.default_rng(SEED)
    for name, size in (("B", rows), ("C", columns)):
        destination = path.with_name(f"{path.stem}_{name}.npy")
        factor = rng.random((size, rank), dtype=np.float32)
        if is_downloaded(destination):
            existing = np.load(destination, allow_pickle=False)
            if (existing.shape != factor.shape or existing.dtype != np.float32
                    or not np.all(np.isfinite(existing))):
                raise ValueError(f"Invalid {operation} factor: {destination}")
            print(f"Already generated: {destination.name}")
            continue

        with tempfile.TemporaryDirectory(prefix=".factor-", dir=MATRICES_DIR) as temp_dir:
            output = Path(temp_dir) / destination.name
            np.save(output, factor, allow_pickle=False)
            output.replace(destination)
        print(f"Generated: {destination}")


def download():
    failed = False
    for matrix in MATRICES:
        try:
            download_matrix(matrix)
            path = matrix_path(matrix)
            if path.suffix == ".mtx":
                generate_vector(path)
            generate_factors(path)
        except (OSError, URLError, tarfile.TarError, EOFError, ValueError) as error:
            print(f"Failed to download {matrix['filename']}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


def clean():
    if MATRICES_DIR.exists():
        script = MATRICES_DIR / "sparse-matrix-data-helper.py"
        for path in MATRICES_DIR.iterdir():
            if path == script:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"Deleted downloaded matrices: {MATRICES_DIR}")
    else:
        print("No downloaded matrices to delete.")
    return 0


def verify():
    missing = False
    for matrix in MATRICES:
        path = matrix_path(matrix)
        paths = [path]
        if path.suffix == ".mtx":
            paths.append(path.with_name(f"{path.stem}_x.npy"))
        paths.extend(path.with_name(f"{path.stem}_{name}.npy") for name in ("B", "C"))
        for candidate in paths:
            if is_downloaded(candidate):
                print(f"OK: {candidate.name}")
            else:
                print(f"Missing or empty: {candidate.name}")
                missing = True
    if not missing:
        print("All matrices, tensors, input vectors, and dense factors are available.")
    return int(missing)


def matrix_market_info(path):
    """Read metadata from the Matrix Market header and file comments."""
    with path.open(encoding="utf-8") as data:
        header = data.readline().lower().split()
        if len(header) != 5 or header[:2] != ["%%matrixmarket", "matrix"]:
            raise ValueError("Invalid Matrix Market header")
        _, _, storage, value_type, symmetry = header
        if storage not in ("coordinate", "array"):
            raise ValueError(f"Unsupported Matrix Market storage: {storage}")
        if value_type not in ("real", "complex", "integer", "pattern"):
            raise ValueError(f"Unsupported Matrix Market value type: {value_type}")
        if symmetry not in ("general", "symmetric", "skew-symmetric", "hermitian"):
            raise ValueError(f"Unsupported Matrix Market symmetry: {symmetry}")
        metadata = {
            "Format": "Matrix Market",
            "Storage": storage,
            "Value type": value_type,
            "Symmetry": symmetry,
        }
        comment_fields = {
            "name": "Name",
            "title": "Description",
            "id": "ID",
            "date": "Date",
            "author": "Author",
            "ed": "Editor",
            "kind": "Kind",
        }
        for line in data:
            line = line.strip()
            if not line:
                continue
            if line.startswith("%"):
                comment = line.lstrip("%").strip()
                key, separator, value = comment.partition(":")
                label = comment_fields.get(key.strip().lower())
                if label and separator and value.strip():
                    metadata[label] = value.strip()
                elif comment.startswith("[") and comment.endswith("]"):
                    metadata["Description"] = comment[1:-1].strip()
                elif comment.startswith(("https://", "http://")):
                    metadata["Source (file comment)"] = comment
                continue
            sizes = [int(value) for value in line.split()]
            expected = 3 if storage == "coordinate" else 2
            if len(sizes) != expected or any(size < 0 for size in sizes):
                raise ValueError("Invalid Matrix Market dimensions or entry count")
            metadata["Dimensions"] = f"{sizes[0]} x {sizes[1]}"
            if storage == "coordinate":
                # Symmetric files may store only one triangle of the matrix.
                metadata["Stored entries (header)"] = sizes[2]
            return metadata
    raise ValueError("Missing Matrix Market dimensions")


def tensor_stats(path):
    """Scan a one-based FROSTT .tns file using memory proportional to its order."""
    dimensions = []
    entries = 0
    with path.open(encoding="utf-8") as data:
        for line_number, line in enumerate(data, start=1):
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Missing tensor coordinates or value on line {line_number}")
            try:
                indices = [int(value) for value in fields[:-1]]
                float(fields[-1])
            except ValueError as error:
                raise ValueError(f"Invalid tensor entry on line {line_number}") from error
            if not dimensions:
                dimensions = [0] * len(indices)
            if len(indices) != len(dimensions) or any(index < 1 for index in indices):
                raise ValueError(f"Invalid tensor coordinates on line {line_number}")
            dimensions = [max(size, index) for size, index in zip(dimensions, indices)]
            entries += 1
    if not entries:
        raise ValueError("No tensor entries found; cannot infer dimensions")
    return dimensions, entries


def tensor_info(path):
    """Format the dimensions and entry count of a FROSTT tensor."""
    dimensions, entries = tensor_stats(path)
    return {
        "Format": "FROSTT coordinate tensor",
        "Order": len(dimensions),
        "Dimensions (inferred from maximum indices)": " x ".join(map(str, dimensions)),
        "Stored entries": entries,
    }


def info():
    failed = False
    for index, matrix in enumerate(MATRICES):
        if index:
            print()
        path = matrix_path(matrix)
        print(path.name)
        print(f"  Download URL: {matrix['url']}")
        print(f"  Local path: {path}")
        if is_downloaded(path):
            print(f"  Status: downloaded ({path.stat().st_size} bytes)")
            try:
                reader = matrix_market_info if path.suffix == ".mtx" else tensor_info
                metadata = reader(path)
            except (OSError, ValueError) as error:
                print(f"Could not read metadata for {path.name}: {error}", file=sys.stderr)
                failed = True
                continue
            for label, value in metadata.items():
                print(f"  {label}: {value}")
        else:
            print("  Status: missing or empty; download to inspect file metadata")
    return int(failed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Download directory: {MATRICES_DIR}",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("download", help="Download all matrices and tensors and generate inputs")
    commands.add_parser("clean", help="Delete downloaded data, preserving this script")
    commands.add_parser("verify", help="Check that every listed matrix exists and is nonempty")
    commands.add_parser("info", help="Read metadata from downloaded matrix and tensor files")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    try:
        actions = {"download": download, "clean": clean, "verify": verify, "info": info}
        return actions[args.command]()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
