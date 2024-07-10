from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Metric:
    name: str
    value: float

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


@dataclass
class Metrics:
    metrics: dict[str, Metric] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.metrics[name] = Metric(name, value)

    def get(self, name: str) -> Metric:
        return self.metrics[name]

    def __getitem__(self, name: str) -> Metric:
        return self.metrics[name]

    def __iter__(self) -> Iterator[Metric]:
        return iter(self.metrics.values())

    def items(self) -> Iterator[tuple[str, Metric]]:
        return self.metrics.items()  # type: ignore

    def __str__(self) -> str:
        return "\n".join(str(metric) for metric in self.metrics.values())
