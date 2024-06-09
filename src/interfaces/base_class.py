from abc import ABC, abstractmethod
from dataclasses import dataclass

from itakello_logging import ItakelloLogging

logger = ItakelloLogging().get_logger(__name__)

@dataclass
class BaseClass(ABC):

    def __post_init__(self) -> None:
        logger.debug(f"Created {self.__class__.__name__} instance.")