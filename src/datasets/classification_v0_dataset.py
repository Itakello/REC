from dataclasses import dataclass, field
from typing import Any

import torch
from itakello_logging import ItakelloLogging

from ..utils.consts import DEVICE
from .refcocog_base_dataset import RefCOCOgBaseDataset

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class ClassificationV0Dataset(RefCOCOgBaseDataset):
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

        # Pad or truncate candidates_embeddings to top_k
        num_candidates = candidates_embeddings.shape[0]
        if num_candidates < self.top_k:
            padding = torch.stack(
                [self.empty_embedding] * (self.top_k - num_candidates)
            )
            candidates_embeddings = torch.cat([candidates_embeddings, padding], dim=0)
        else:
            candidates_embeddings = candidates_embeddings[: self.top_k]

        item.update(
            {
                "candidates_embeddings": candidates_embeddings,
                "combined_sentences_embedding": combined_sentences_embedding,
                "correct_candidate_idx": torch.tensor(
                    row["ordered_correct_candidate_idx"]
                ).int(),
            }
        )

        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {
            "candidates_embeddings": [],
            "combined_sentences_embeddings": [],
            "correct_candidate_indices": [],
        }

        for item in batch:
            collated_batch["candidates_embeddings"].append(
                item["candidates_embeddings"]
            )
            collated_batch["combined_sentences_embeddings"].append(
                item["combined_sentences_embedding"]
            )
            collated_batch["correct_candidate_indices"].append(
                item["correct_candidate_idx"]
            )

        collated_batch["candidates_embeddings"] = torch.stack(  # type: ignore
            collated_batch["candidates_embeddings"]
        )
        collated_batch["combined_sentences_embeddings"] = torch.stack(  # type: ignore
            collated_batch["combined_sentences_embeddings"]
        )
        collated_batch["correct_candidate_indices"] = torch.stack(  # type: ignore
            collated_batch["correct_candidate_indices"]
        )

        return collated_batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Initialize the dataset
    dataset = ClassificationV0Dataset(split="train", top_k=6)

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

    # Print the first item's correct candidate index
    print(
        f"First item's correct candidate index: {batch['correct_candidate_indices'][0]}"
    )

    logger.info("ClassificationV0Dataset test completed successfully")
