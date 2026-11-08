import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass
class BenchmarkResult:
    run_utc: str
    backend: str
    operation: str
    operand_names: list[str]
    device: int
    warmup: int
    samples_ms: list[float]
    timing: str = "cuda_events"
    scope: str = "execution"

    def __post_init__(self):
        assert self.samples_ms
        assert isinstance(self.operand_names, list)
        assert self.operand_names
        for name in self.operand_names:
            assert isinstance(name, str)
            assert name
            assert Path(name).name == name
            assert ";" not in name

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
        details = list(self.operand_names)
        details.append(f"{median_ms=:.6f}, {samples=}")
        print(f"{self.backend} {self.operation.upper()}: {', '.join(details)}")

    def write_csv(self) -> None:
        """Save one CSV per operation and operands, replacing the previous run."""
        row = {
            "run_utc": self.run_utc,
            "backend": self.backend,
            "operation": self.operation,
            "operand_names": ";".join(self.operand_names),
            "device": self.device,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "timing": self.timing,
            "scope": self.scope,
        }
        fieldnames = [*row, "iteration", "elapsed_ms"]

        operands = [Path(name).stem for name in self.operand_names]
        filename = "__".join([self.operation, *operands]) + ".csv"
        path = Path("generated") / self.backend / "results" / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for iteration, elapsed in enumerate(self.samples_ms, start=1):
                writer.writerow({**row, "iteration": iteration, "elapsed_ms": float(elapsed)})
