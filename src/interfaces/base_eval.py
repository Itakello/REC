from abc import ABC, abstractmethod
from dataclasses import dataclass

from itakello_logging import ItakelloLogging

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class BaseEvaluator(ABC):
    eval_name: str

    def __post_init__(self) -> None:
        logger.debug(f"Initializing [{self.eval_name}] evaluator.")

    @abstractmethod
    def evaluate(self) -> None:
        pass

    @abstractmethod
    def get_dataloaders(self) -> list[tuple]:
        pass
