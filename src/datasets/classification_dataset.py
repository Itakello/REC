import json
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from itakello_logging import ItakelloLogging

from ..utils.consts import DEVICE
from .refcocog_base_dataset import RefCOCOgBaseDataset

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class ClassificationDataset(RefCOCOgBaseDataset):
    top_k: int = 6
    empty_embedding: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.empty_embedding = torch.zeros(1024).to(DEVICE)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        row = self.data.iloc[index]

        embeddings = self.get_embeddings(row["embeddings_filename"])

        candidates_embeddings = embeddings["candidates_embeddings"]
        combined_sentences_embedding = embeddings["combined_sentences_features"]

        # Get the ordered indices of candidates
        ordered_indices = json.loads(row["ordered_candidate_indices"])

        if isinstance(ordered_indices, int):
            ordered_indices = [ordered_indices]

        # Select top-k candidates based on ordered indices
        top_k_indices = ordered_indices[: self.top_k]
        candidates_embeddings = candidates_embeddings[top_k_indices]

        # Pad if necessary
        num_candidates = candidates_embeddings.shape[0]
        if num_candidates < self.top_k:
            padding = torch.stack(
                [self.empty_embedding] * (self.top_k - num_candidates)
            )
            candidates_embeddings = torch.cat([candidates_embeddings, padding], dim=0)

        correct_candidate_idx = row["ordered_correct_candidate_idx"]

        # Create target with an extra class for "no correct candidate"
        if correct_candidate_idx == -1 or correct_candidate_idx > self.top_k:
            correct_candidate_idx = self.top_k

        target = (
            F.one_hot(torch.tensor(correct_candidate_idx), num_classes=self.top_k + 1)
            .float()
            .to(DEVICE)
        )

        item.update(
            {
                "candidates_embeddings": candidates_embeddings,
                "combined_sentences_embedding": combined_sentences_embedding,
                "target": target,
            }
        )

        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {
            "candidates_embeddings": [],
            "combined_sentences_embeddings": [],
            "targets": [],
        }

        for item in batch:
            collated_batch["candidates_embeddings"].append(
                item["candidates_embeddings"]
            )
            collated_batch["combined_sentences_embeddings"].append(
                item["combined_sentences_embedding"]
            )
            collated_batch["targets"].append(item["target"])

        collated_batch["candidates_embeddings"] = torch.stack(  # type: ignore
            collated_batch["candidates_embeddings"]
        )
        collated_batch["combined_sentences_embeddings"] = torch.stack(  # type: ignore
            collated_batch["combined_sentences_embeddings"]
        )
        collated_batch["targets"] = torch.stack(collated_batch["targets"])  # type: ignore

        return collated_batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Initialize the dataset
    dataset = ClassificationDataset(split="train", top_k=6)

    print(f"Dataset length: {len(dataset)}")

    # Create a DataLoader
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn
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

    logger.info("ClassificationV0Dataset test completed successfully")
