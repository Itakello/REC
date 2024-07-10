from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch.utils.data import DataLoader

from ..classes.metric import Metrics
from .base_class import BaseClass


@dataclass
class BaseEval(BaseClass, ABC):
    name: str
    run_name: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    @abstractmethod
    def evaluate(self) -> Metrics | dict[str, Metrics]:
        pass

    @abstractmethod
    def get_dataloaders(self) -> list[tuple[str, DataLoader]]:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Metrics | dict[str, Metrics]) -> None:
        pass
