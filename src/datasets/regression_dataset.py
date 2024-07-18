from dataclasses import dataclass, field
from typing import Any

import torch
from itakello_logging import ItakelloLogging
from PIL import Image

from ..classes.highlighting_modality import HighlightingModality
from ..models.clip_model import ClipModel
from ..utils.consts import CLIP_MODEL, DEVICE
from .refcocog_base_dataset import RefCOCOgBaseDataset

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class RegressionDataset(RefCOCOgBaseDataset):
    top_k: int = 6
    empty_embedding: torch.Tensor = field(init=False)
    clip: ClipModel = field(default_factory=ClipModel)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.empty_embedding = torch.zeros(1024).to(DEVICE)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        row = self.data.iloc[index]

        embeddings = self.get_embeddings(
            row["embeddings_filename"]
        )  # all offline embeddings
        reordered_candidates_embeddings = embeddings[
            "candidates_embeddings"
        ]  # embeddings of all candidates (not ordered)
        combined_sentences_embedding = embeddings["combined_sentences_features"]
        image_embedding = embeddings["image_features"]

        # Get the ordered indices of candidates
        ordered_indices = self.get_bbox(row["ordered_candidate_indices"])

        if isinstance(ordered_indices, int):
            ordered_indices = [ordered_indices]

        # Select top-k candidates based on ordered indices
        top_k_indices = ordered_indices[: self.top_k]
        reordered_candidates_embeddings = reordered_candidates_embeddings[top_k_indices]

        # Pad if necessary
        num_candidates = reordered_candidates_embeddings.shape[0]
        if num_candidates < self.top_k:
            padding = torch.stack(
                [self.empty_embedding] * (self.top_k - num_candidates)
            )
            reordered_candidates_embeddings = torch.cat(
                [reordered_candidates_embeddings, padding], dim=0
            )

        candidates_bounding_boxes = self.get_bbox(row["yolo_predictions"])

        reordered_bounding_boxes = []

        # Reorder the bounding boxes based on the importance of the candidates
        for original_idx in top_k_indices:
            reordered_bounding_boxes.append(
                torch.Tensor(candidates_bounding_boxes[original_idx])
            )

        reordered_bounding_boxes = torch.stack(reordered_bounding_boxes).to(DEVICE)

        if num_candidates < self.top_k:
            # Pad with the default bounding box
            padding = torch.zeros((self.top_k - num_candidates, 4), device=DEVICE)
            reordered_bounding_boxes = torch.cat(
                [reordered_bounding_boxes, padding], dim=0
            ).to(DEVICE)

        image_path = self.get_image_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")
        gold_bbox = torch.Tensor(self.get_bbox(row["bbox"]))
        gold_embedding = self.clip.encode_images(
            HighlightingModality.apply_highlighting(image, gold_bbox, "crop")
        )

        item.update(
            {
                "image": image,
                "candidates_embeddings": reordered_candidates_embeddings,
                "combined_sentence_embeddings": combined_sentences_embedding,
                "candidates_bboxes": reordered_bounding_boxes,
                "image_embedding": image_embedding,
                "gold_bbox": gold_bbox,
                "gold_embedding": gold_embedding,
            }
        )

        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {
            "candidates_embeddings": [],
            "combined_sentences_embeddings": [],
            "images": [],
            "candidates_bboxes": [],
            "image_embeddings": [],
            "gold_bboxes": [],
            "gold_embeddings": [],
        }

        for item in batch:
            collated_batch["images"].append(item["image"])
            collated_batch["candidates_embeddings"].append(
                item["candidates_embeddings"]
            )
            collated_batch["combined_sentences_embeddings"].append(
                item["combined_sentence_embeddings"]
            )
            collated_batch["candidates_bboxes"].append(item["candidates_bboxes"])
            collated_batch["image_embeddings"].append(item["image_embedding"])
            collated_batch["gold_bboxes"].append(item["gold_bbox"])
            collated_batch["gold_embeddings"].append(item["gold_embedding"])

        collated_batch["candidates_embeddings"] = torch.stack(  # type: ignore
            collated_batch["candidates_embeddings"]
        )
        collated_batch["combined_sentences_embeddings"] = torch.stack(  # type: ignore
            collated_batch["combined_sentences_embeddings"]
        )
        collated_batch["candidates_bboxes"] = torch.stack(  # type: ignore
            collated_batch["candidates_bboxes"]
        )
        collated_batch["image_embeddings"] = torch.stack(  # type: ignore
            collated_batch["image_embeddings"]
        )
        collated_batch["gold_bboxes"] = torch.stack(  # type: ignore
            collated_batch["gold_bboxes"]
        )
        collated_batch["gold_embeddings"] = torch.stack(  # type: ignore
            collated_batch["gold_embeddings"]
        )

        return collated_batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Initialize the dataset
    dataset = RegressionDataset(
        split="train", top_k=6, clip=ClipModel(version=CLIP_MODEL)
    )

    print(f"Dataset length: {len(dataset)}")

    # Create a DataLoader
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=True, collate_fn=dataset.collate_fn
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Print the shapes of the batch elements
    print("Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    logger.info("RegressionDataset test completed successfully")
