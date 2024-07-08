from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from itakello_logging import ItakelloLogging
from torch.utils.data import Dataset

from .base_class import BaseClass

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class BaseDataset(BaseClass, Dataset, ABC):
    split: str = "train"
    limit: int = -1
    data: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.load_data()
        self.data = self.data[self.data["split"] == self.split]
        logger.info(f"Loaded {len(self.data)} {self.split} samples")
        if self.limit != -1:
            self.data = self.data.sample(
                n=min(self.limit, len(self.data)), random_state=42
            )
            logger.info(f"Limited to {len(self.data)} samples")

    @abstractmethod
    def load_data(self) -> None:
        """
        Load the dataset from a source (e.g., file, database).
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get an item from the dataset at the specified index.
        This method should be implemented by subclasses.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the item data.
        """
        pass

    def __len__(self) -> int:
        """
        Get the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)
