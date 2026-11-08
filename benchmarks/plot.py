"""Plot elapsed time versus benchmark case using vertical box plots.

By default, read the current results from generated/*/results/*.csv.
Operations, devices, timing methods, and timing scopes get separate figures.
Each figure is saved as PNG and SVG in generated/plots.
Filenames include the system name alongside the operation and timing context.

Examples:
    python -m benchmarks.plot
    python -m benchmarks.plot generated/cusparse/results/spmv_*.csv --log
    python -m benchmarks.plot --operation spmv
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from .result import BenchmarkResult


_REQUIRED_COLUMNS = {
    "run_utc", "backend", "operation", "operand_names", "device", "warmup",
    "repeat", "timing", "scope", "iteration", "elapsed_ms",
}


def case_key(result: BenchmarkResult) -> tuple:
    """Keep every input and the measurement context in the case identity."""
    return (
        result.backend, result.operation, tuple(result.operand_names),
        result.device, result.timing, result.scope,
    )


def result_from_row(row: dict[str, str]) -> BenchmarkResult:
    elapsed = float(row["elapsed_ms"])
    assert math.isfinite(elapsed)
    assert elapsed >= 0

    timestamp = datetime.fromisoformat(row["run_utc"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    for name in ("backend", "operation", "timing", "scope"):
        assert row[name].strip()

    return BenchmarkResult(
        run_utc=timestamp.astimezone(timezone.utc).isoformat(),
        backend=row["backend"],
        operation=row["operation"],
        operand_names=row["operand_names"].split(";"),
        device=int(row["device"]),
        warmup=int(row["warmup"]),
        samples_ms=[elapsed],
        timing=row["timing"],
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
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("Row does not match the CSV header")
                    result = result_from_row(row)
                except (AssertionError, KeyError, ValueError) as error:
                    raise ValueError(f"{path}:{reader.line_num}: {error}") from error
                key = (case_key(result), result.run_utc, result.warmup)
                if key in runs:
                    runs[key].samples_ms.extend(result.samples_ms)
                else:
                    runs[key] = result
    return list(runs.values())


def plot_results(
    results: list[BenchmarkResult], output_dir: Path, *, log_scale: bool = False,
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
            tuple(result.operand_names),
            result.backend, result.run_utc,
        ))
        backends = sorted({result.backend for result in runs})
        labels = []
        for result in runs:
            label = "\n".join(result.operand_names)
            if len(backends) > 1:
                label += f"\n{result.backend}"
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
    parser.add_argument("csv", type=Path, nargs="*", help="Input CSVs (default: generated/*/results/*.csv)")
    parser.add_argument("--operation", help="Only plot this operation, e.g. spmv or spgemm")
    parser.add_argument("--output-dir", type=Path, default=Path("generated/plots"))
    parser.add_argument("--log", action="store_true", help="Use a logarithmic elapsed-time axis")
    args = parser.parse_args()
    paths = args.csv or sorted(Path("generated").glob("*/results/*.csv"))
    if not paths:
        parser.exit(1, "No benchmark CSVs found. Run a benchmark first or provide CSV paths.\n")
    try:
        results = load_results(paths)
        if args.operation:
            results = [result for result in results if result.operation == args.operation]
        outputs = plot_results(results, args.output_dir, log_scale=args.log)
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    systems = ", ".join(sorted({result.backend for result in results}))
    print(f"Plotted {len(results)} runs from {systems}; saved {len(outputs)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
