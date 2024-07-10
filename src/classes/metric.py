from dataclasses import dataclass, field
from typing import Any, Dict, Iterator

from _collections_abc import dict_items


@dataclass
class Metric:
    name: str
    value: float

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


@dataclass
class Metrics:
    metrics: Dict[str, Metric] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.metrics[name] = Metric(name, value)

    def get(self, name: str) -> Metric:
        return self.metrics[name]

    def __getitem__(self, name: str) -> Metric:
        return self.metrics[name]

    def __iter__(self) -> Iterator[Metric]:
        return iter(self.metrics.values())

    def items(self) -> dict_items[str, Metric]:
        return self.metrics.items()

    def __str__(self) -> str:
        return "\n".join(str(metric) for metric in self.metrics.values())
