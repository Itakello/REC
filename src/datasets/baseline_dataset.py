from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..datasets.refcocog_base_dataset import RefCOCOgBaseDataset
from ..utils.consts import DEVICE


@dataclass
class BaselineDataset(RefCOCOgBaseDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        embeddings = self.get_embeddings(row["embeddings_filename"])

        return {
            "candidates_embeddings": embeddings["candidates_embeddings"],
            "sentence_embedding": embeddings["combined_sentences_features"],
            "gt_bbox": self.get_bbox((row["bbox"])),
            "yolo_predictions": self.get_bbox(row["yolo_predictions"]),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {
            "candidates_embeddings": [],
            "sentence_embeddings": [],
            "gt_bboxes": [],
            "yolo_predictions": [],
        }
        for item in batch:
            collated_batch["candidates_embeddings"].append(
                item["candidates_embeddings"]
            )
            collated_batch["sentence_embeddings"].append(item["sentence_embedding"])
            collated_batch["gt_bboxes"].append(item["gt_bbox"])
            collated_batch["yolo_predictions"].append(item["yolo_predictions"])

        collated_batch["sentence_embeddings"] = torch.stack(  # type: ignore
            collated_batch["sentence_embeddings"]
        )
        collated_batch["gt_bboxes"] = torch.tensor(collated_batch["gt_bboxes"])  # type: ignore

        return collated_batch
