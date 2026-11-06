"""Plot elapsed time versus benchmark case using vertical box plots.

By default, read generated/*/*.csv and plot the latest run per input/backend.
Operations, devices, timing methods, and timing scopes get separate figures.
Each figure is saved as PNG and SVG in generated/plots.
Filenames include the system name alongside the operation and timing context.

Examples:
    python -m benchmarks.plot
    python benchmarks/plot.py generated/cusparse/spmv.csv --log
    python -m benchmarks.plot --operation spmv --all-runs
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from .result import BenchmarkResult, InputInfo


ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_COLUMNS = {
    "run_utc", "backend", "operation", "device", "warmup", "scope", "elapsed_ms",
}


def case_key(result: BenchmarkResult) -> tuple:
    """Keep every input and the measurement context in the case identity."""
    inputs = tuple(
        (name, info.path, info.shape, info.nnz)
        for name, info in sorted(result.inputs.items())
    )
    return (result.backend, result.operation, inputs, result.device, result.timing, result.scope)


def result_from_row(row: dict[str, str]) -> BenchmarkResult:
    elapsed = float(row["elapsed_ms"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_ms must be finite and nonnegative")

    timestamp = datetime.fromisoformat(row["run_utc"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    inputs = {}
    if "a_path" in row:
        for name in ("A", "B", "x"):
            prefix = name.lower()
            path = row.get(f"{prefix}_path")
            if not path:
                continue
            shape = (int(row[f"{prefix}_rows"]),)
            if row.get(f"{prefix}_cols"):
                shape += (int(row[f"{prefix}_cols"]),)
            nnz = row.get(f"{prefix}_nnz")
            inputs[name] = InputInfo(path, shape, int(nnz) if nnz else None)
    else:
        # The existing Nacho runner predates BenchmarkResult's column names.
        inputs = {
            "A": InputInfo(row["matrix_a"], (int(row["m"]), int(row["k"])), int(row["nnz_a"])),
            "B": InputInfo(row["matrix_b"], (int(row["k"]), int(row["n"])), int(row["nnz_b"])),
        }
    if "A" not in inputs or not inputs["A"].path:
        raise ValueError("Missing input A")
    for name in ("backend", "operation", "scope"):
        if not row[name].strip():
            raise ValueError(f"Missing {name}")

    return BenchmarkResult(
        run_utc=timestamp.astimezone(timezone.utc).isoformat(),
        backend=row["backend"],
        operation=row["operation"],
        inputs=inputs,
        device=int(row["device"]),
        warmup=int(row["warmup"]),
        samples_ms=[elapsed],
        timing=row.get("timing") or "unspecified",
        scope=row["scope"],
    )


def load_results(paths: list[Path]) -> list[BenchmarkResult]:
    runs = {}
    for path in dict.fromkeys(path.resolve() for path in paths):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = _REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
            if not {"a_path", "matrix_a"} & columns:
                raise ValueError(f"{path}: expected an a_path or matrix_a input column")
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("Row does not match the CSV header")
                    result = result_from_row(row)
                except (KeyError, ValueError) as error:
                    raise ValueError(f"{path}:{reader.line_num}: {error}") from error
                key = (case_key(result), result.run_utc, result.warmup)
                if key in runs:
                    runs[key].samples_ms.extend(result.samples_ms)
                else:
                    runs[key] = result
    return list(runs.values())


def latest_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    latest = {}
    for result in results:
        key = case_key(result)
        if key not in latest or result.run_utc > latest[key].run_utc:
            latest[key] = result
    return list(latest.values())


def plot_results(
    results: list[BenchmarkResult], output_dir: Path, *, all_runs: bool = False, log_scale: bool = False,
) -> list[Path]:
    if not results:
        raise ValueError("No benchmark results to plot")
    if log_scale and any(sample <= 0 for result in results for sample in result.samples_ms):
        raise ValueError("A logarithmic axis requires positive timings; use the default linear axis for zeros")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install the plotting dependency: python -m pip install matplotlib==3.11.2") from error

    groups = defaultdict(list)
    for result in results:
        groups[(result.operation, result.device, result.timing, result.scope)].append(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for (operation, device, timing, scope), runs in sorted(groups.items()):
        runs.sort(key=lambda result: (
            tuple((name, info.path) for name, info in sorted(result.inputs.items())),
            result.backend, result.run_utc,
        ))
        backends = sorted({result.backend for result in runs})
        labels = []
        for result in runs:
            label = "\n".join(
                f"{name}={Path(info.path).name}" for name, info in sorted(result.inputs.items())
            )
            if len(backends) > 1:
                label += f"\n{result.backend}"
            if all_runs:
                label += f"\n{result.run_utc}"
            labels.append(label)

        figure, axis = plt.subplots(figsize=(max(7, len(runs) * 2), 5), layout="constrained")
        axis.boxplot(
            [result.samples_ms for result in runs],
            orientation="vertical", tick_labels=labels,
        )
        axis.set_title(f"System: {', '.join(backends)} — {operation.upper()}")
        axis.set_xlabel("Case")
        axis.set_ylabel("Elapsed time (ms)")
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)
        name = re.sub(
            r"[^A-Za-z0-9_.-]+", "_",
            f"{'_'.join(backends)}_{operation}_device{device}_{timing}_{scope}",
        )
        for extension in ("png", "svg"):
            path = output_dir / f"{name}.{extension}"
            figure.savefig(path, dpi=180)
            outputs.append(path)
        plt.close(figure)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="*", help="Input CSVs (default: generated/*/*.csv)")
    parser.add_argument("--operation", help="Only plot this operation, e.g. spmv or spgemm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated" / "plots")
    parser.add_argument("--all-runs", action="store_true", help="Show each historical run as a separate box")
    parser.add_argument("--log", action="store_true", help="Use a logarithmic elapsed-time axis")
    args = parser.parse_args()
    paths = args.csv or sorted((ROOT / "generated").glob("*/*.csv"))
    if not paths:
        parser.exit(1, "No benchmark CSVs found. Run a benchmark first or provide CSV paths.\n")
    try:
        results = load_results(paths)
        if args.operation:
            results = [result for result in results if result.operation == args.operation]
        if not args.all_runs:
            results = latest_results(results)
        outputs = plot_results(results, args.output_dir, all_runs=args.all_runs, log_scale=args.log)
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    systems = ", ".join(sorted({result.backend for result in results}))
    print(f"Plotted {len(results)} runs from {systems}; saved {len(outputs)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
