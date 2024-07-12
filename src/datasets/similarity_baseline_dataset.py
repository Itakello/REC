from dataclasses import dataclass
from typing import Any

import torch

from .yolo_baseline_dataset import YOLOBaselineDataset


@dataclass
class SimilarityBaselineDataset(YOLOBaselineDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        row = self.data.iloc[index]

        embeddings = self.get_embeddings(
            row["embeddings_filename"],
            [
                "original_sentences_features",
                "comprehensive_sentence_features",
                "combined_sentences_features",
            ],
        )

        item.update(
            {
                "original_sentences_embeddings": embeddings[
                    "original_sentences_features"
                ],
                "comprehensive_sentence_embedding": embeddings[
                    "comprehensive_sentence_features"
                ],
                "combined_sentences_embeddings": embeddings[
                    "combined_sentences_features"
                ],
            }
        )

        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = YOLOBaselineDataset.collate_fn(batch)

        collated_batch.update(
            {
                "original_sentences_embeddings": [],
                "comprehensive_sentence_embeddings": [],
                "combined_sentences_embeddings": [],
            }
        )

        for item in batch:
            collated_batch["original_sentences_embeddings"].append(
                item["original_sentences_embeddings"]
            )
            collated_batch["comprehensive_sentence_embeddings"].append(
                item["comprehensive_sentence_embedding"]
            )
            collated_batch["combined_sentences_embeddings"].append(
                item["combined_sentences_embeddings"]
            )

        collated_batch["original_sentences_embeddings"] = torch.stack(
            collated_batch["original_sentences_embeddings"]
        )
        collated_batch["comprehensive_sentence_embeddings"] = torch.stack(
            collated_batch["comprehensive_sentence_embeddings"]
        )
        collated_batch["combined_sentences_embeddings"] = torch.stack(
            collated_batch["combined_sentences_embeddings"]
        )

        return collated_batch


if __name__ == "__main__":
    from pprint import pprint

    from ..utils.consts import DATA_PATH

    dataset = SimilarityBaselineDataset(
        annotations_path=DATA_PATH / "annotations.csv",
        images_path=DATA_PATH / "images",
        embeddings_path=DATA_PATH / "embeddings",
    )

    print(f"Dataset length: {len(dataset)}")

    # Get the first item
    first_item = dataset[0]

    # Display the image
    first_item["image"].show()

    # Create a copy of the dictionary without the image for printing
    printable_item = {k: v for k, v in first_item.items() if k != "image"}

    # Pretty print the dictionary
    print("First item contents:")
    pprint(printable_item)
