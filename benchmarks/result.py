import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


_CSV_INPUT_NAMES = ("A", "B", "x")


@dataclass
class InputInfo:
    path: str
    shape: tuple[int, ...]
    nnz: int | None = None


@dataclass
class BenchmarkResult:
    run_utc: str
    backend: str
    operation: str
    inputs: dict[str, InputInfo]
    device: int
    warmup: int
    samples_ms: list[float]
    timing: str = "cuda_events"
    scope: str = "execution"

    def __post_init__(self):
        if not self.samples_ms:
            raise ValueError("A benchmark result must contain at least one sample")

    @property
    def repeat(self) -> int:
        return len(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return float(median(self.samples_ms))

    @property
    def mean_ms(self) -> float:
        return float(mean(self.samples_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.samples_ms))

    @property
    def max_ms(self) -> float:
        return float(max(self.samples_ms))

    def print_summary(self) -> None:
        median_ms = self.median_ms
        samples = self.repeat
        details = [f"{name}={Path(info.path).name}" for name, info in self.inputs.items()]
        details.append(f"{median_ms=:.6f}, {samples=}")
        print(f"{self.backend} {self.operation.upper()}: {', '.join(details)}")

    def write_csv(self, path: str | Path) -> None:
        """Append one row per sample using a shared schema for A, B, and x."""
        unknown_inputs = self.inputs.keys() - set(_CSV_INPUT_NAMES)
        if unknown_inputs:
            raise ValueError(f"Unsupported CSV input names: {sorted(unknown_inputs)}")

        fieldnames = ["run_utc", "backend", "operation"]
        row = {
            "run_utc": self.run_utc,
            "backend": self.backend,
            "operation": self.operation,
            "device": self.device,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "timing": self.timing,
            "scope": self.scope,
        }
        for name in _CSV_INPUT_NAMES:
            prefix = name.lower()
            fieldnames.extend(f"{prefix}_{field}" for field in ("path", "rows", "cols", "nnz"))
            info = self.inputs.get(name)
            if info is None:
                continue
            if len(info.shape) not in (1, 2):
                raise ValueError(f"CSV inputs must be vectors or matrices: {name} has shape {info.shape}")
            row.update({
                f"{prefix}_path": info.path,
                f"{prefix}_rows": info.shape[0],
                f"{prefix}_cols": info.shape[1] if len(info.shape) == 2 else None,
                f"{prefix}_nnz": info.nnz,
            })
        fieldnames.extend(("device", "warmup", "repeat", "timing", "scope", "iteration", "elapsed_ms"))

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", newline="", encoding="utf-8") as output:
            output.seek(0)
            header = next(csv.reader(output), None)
            if header is not None and header != fieldnames:
                raise ValueError(
                    f"Incompatible CSV header in {path}; move or rename the existing file "
                    "before writing results with the new schema"
                )
            output.seek(0, 2)
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            if header is None:
                writer.writeheader()
            for iteration, elapsed in enumerate(self.samples_ms, start=1):
                writer.writerow({**row, "iteration": iteration, "elapsed_ms": float(elapsed)})
