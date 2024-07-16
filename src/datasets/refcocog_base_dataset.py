import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from itakello_logging import ItakelloLogging
from PIL import Image
from torch.utils.data import DataLoader

from ..interfaces.base_dataset import BaseDataset
from ..utils.consts import (
    DEVICE,
    EMBEDDINGS_PATH,
    IMAGES_PATH,
    PROCESSED_ANNOTATIONS_PATH,
)

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class RefCOCOgBaseDataset(BaseDataset):
    split: str | None = None
    limit: int = -1
    data: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._load_data()
        self._filter_data()

    def _load_data(self) -> None:
        self.data = pd.read_csv(PROCESSED_ANNOTATIONS_PATH / "annotations.csv")
        logger.info(f"Loaded {len(self.data)} total samples")

    def _filter_data(self) -> None:
        if self.split is not None:
            self.data = self.data[self.data["split"] == self.split]
            logger.info(f"Filtered to {len(self.data)} {self.split} samples")
        if self.limit > 0:
            self.data = self.data.sample(
                n=min(self.limit, self.data.__len__()), random_state=42
            )
            logger.info(f"Limited to {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def get_image_path(self, file_name: str) -> Path:
        return IMAGES_PATH / file_name

    def get_bbox(self, bbox_str: str) -> list:
        return json.loads(bbox_str)

    def get_embeddings(
        self, embeddings_filename: str, keys: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        full_path = EMBEDDINGS_PATH / embeddings_filename
        embeddings = np.load(full_path)

        if keys is None:
            return {k: embeddings[k] for k in embeddings.keys()}

        result = {}
        for key in keys:
            if key not in embeddings:
                raise KeyError(f"Embedding key '{key}' not found in {full_path}")
            result[key] = torch.from_numpy(embeddings[key]).to(DEVICE)

        return result

    def get_sentences(self, sentences: str) -> list[str]:
        return ast.literal_eval(sentences)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]

        image_path = self.get_image_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")
        bbox = self.get_bbox(row["bbox"])
        embeddings = self.get_embeddings(row["embeddings_filename"])
        sentences = self.get_sentences(row["sentences"])

        return {
            "image": image,
            "bbox": bbox,
            "file_name": row["file_name"],
            "embeddings": embeddings,
            "sentences": sentences,
            "comprehensive_sentence": row["comprehensive_sentence"],
            "category": row["category"],
            "supercategory": row["supercategory"],
            "area": row["area"],
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {
            "images": [],
            "bboxes": [],
            "embeddings": [],
            "file_names": [],
            "sentences": [],
            "comprehensive_sentences": [],
            "categories": [],
            "supercategories": [],
            "areas": [],
        }
        for item in batch:
            collated_batch["images"].append(item["image"])
            collated_batch["bboxes"].append(torch.tensor(item["bbox"]))
            collated_batch["embeddings"].append(item["embeddings"])
            collated_batch["file_names"].append(item["file_name"])
            collated_batch["sentences"].append(item["sentences"])
            collated_batch["comprehensive_sentences"].append(
                item["comprehensive_sentence"]
            )
            collated_batch["categories"].append(item["category"])
            collated_batch["supercategories"].append(item["supercategory"])
            collated_batch["areas"].append(item["area"])

        collated_batch["bboxes"] = torch.stack(collated_batch["bboxes"])  # type: ignore

        return collated_batch

    @classmethod
    def get_dataloaders(
        cls, batch_size: int = 32, limit: int = -1
    ) -> dict[str, DataLoader]:
        dataloaders = {
            "train": DataLoader(
                cls(split="train", limit=limit),
                batch_size=batch_size,
                collate_fn=cls.collate_fn,
                shuffle=True,
            ),
            "val": DataLoader(
                cls(split="val", limit=limit),
                batch_size=batch_size,
                collate_fn=cls.collate_fn,
            ),
            "test": DataLoader(
                cls(split="test", limit=limit),
                batch_size=batch_size,
                collate_fn=cls.collate_fn,
            ),
        }
        return dataloaders


if __name__ == "__main__":

    from pprint import pprint

    dataset = RefCOCOgBaseDataset()

    print(f"Dataset length: {len(dataset)}")

    # Get the first item
    first_item = dataset[0]

    # Display the image
    first_item["image"].show()

    # Create a copy of the dictionary without the image for printing
    printable_item = {k: v for k, v in first_item.items() if k != "image"}

    # Pretty print the dictionary
    print("First item contents:")
    pprint(printable_item, depth=2, compact=True)
