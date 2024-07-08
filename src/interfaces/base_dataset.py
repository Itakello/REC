from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class BaseDataset(Dataset, ABC):
    def __init__(self, data_path: Path, split: str, transform: Compose = None):
        self.data_path = data_path
        self.split = split
        self.images_path = data_path / "images"
        self.annotations_path = data_path / "annotations"
        self.transform = transform or Compose([Resize((224, 224)), ToTensor()])
        self.data: pd.DataFrame = pd.DataFrame()
        self.load_data()

    @abstractmethod
    def load_data(self) -> None:
        """Load data from files and populate self.data DataFrame"""
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """
        Return a tuple of (image, sentence, bounding_box)

        image: torch.Tensor representing the image
        sentence: str representing the referring expression
        bounding_box: torch.Tensor of shape (4,) representing [x1, y1, x2, y2]
        """
        row = self.data.iloc[idx]
        image_path = self.images_path / f"{row['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, row["sentence"], row["bbox"]

    def __len__(self) -> int:
        return len(self.data)
