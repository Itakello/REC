from abc import abstractmethod
from dataclasses import dataclass

from torch.utils.data import Dataset

from .base_class import BaseClass


@dataclass
class BaseDataset(BaseClass, Dataset):

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
