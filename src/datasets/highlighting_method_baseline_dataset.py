from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from ..utils.calculate_iou import calculate_iou
from ..utils.consts import DEVICE
from .refcocog_base_dataset import RefCOCOgBaseDataset


@dataclass
class HighlightingMethodBaselineDataset(RefCOCOgBaseDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]

        image_path = self.get_image_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")

        yolo_predictions = torch.tensor(eval(row["yolo_predictions"])).to(DEVICE)
        gt_bbox = torch.tensor(self.get_bbox(row["bbox"])).to(DEVICE)

        embeddings = self.get_embeddings(
            row["embeddings_filename"], ["combined_sentences_features"]
        )

        return {
            "image": image,
            "yolo_predictions": yolo_predictions,
            "gt_bbox": gt_bbox,
            "correct_candidate_idx": row["correct_candidate_idx"],
            "combined_sentences_embedding": embeddings["combined_sentences_features"],
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {
            "images": [],
            "yolo_predictions": [],
            "gt_bboxes": [],
            "correct_candidates_idx": [],
            "combined_sentences_embeddings": [],
        }

        for item in batch:
            collated_batch["images"].append(item["image"])
            collated_batch["yolo_predictions"].append(item["yolo_predictions"])
            collated_batch["gt_bboxes"].append(item["gt_bbox"])
            collated_batch["correct_candidates_idx"].append(
                item["correct_candidate_idx"]
            )
            collated_batch["combined_sentences_embeddings"].append(
                item["combined_sentences_embedding"]
            )

        return collated_batch

    def filter_valid_samples(self, iou_threshold: float = 0.5) -> None:
        valid_indices = []
        for index, row in self.data.iterrows():
            yolo_predictions = torch.tensor(eval(row["yolo_predictions"]))
            gt_bbox = torch.tensor(self.get_bbox(row["bbox"]))

            # Check if any prediction matches the ground truth
            valid = any(
                calculate_iou(pred, gt_bbox)[0] >= iou_threshold
                for pred in yolo_predictions
            )
            if valid:
                valid_indices.append(index)

        self.data = self.data.loc[valid_indices]
        print(f"Filtered dataset to {len(self.data)} valid samples")


if __name__ == "__main__":
    dataset = HighlightingMethodBaselineDataset(split="val")
    dataset.filter_valid_samples(iou_threshold=0.8)
    print(f"Dataset length: {len(dataset)}")

    # Get the first item
    first_item = dataset[0]

    # Display the image
    first_item["image"].show()

    # Print other information
    print("YOLO predictions shape:", first_item["yolo_predictions"].shape)
    print("Ground truth bbox:", first_item["gt_bbox"])
    print(
        "Combined sentences embedding shape:",
        first_item["combined_sentences_embedding"].shape,
    )
