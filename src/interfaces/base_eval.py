from abc import abstractmethod
from dataclasses import dataclass

from itakello_logging import ItakelloLogging

from .base_class import BaseClass

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class BaseEvaluator(BaseClass):

    @abstractmethod
    def evaluate(self) -> None:
        pass

    @abstractmethod
    def get_dataloaders(self) -> list[tuple]:
        pass
