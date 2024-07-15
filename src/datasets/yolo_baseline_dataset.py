from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .refcocog_base_dataset import RefCOCOgBaseDataset


@dataclass
class YOLOBaselineDataset(RefCOCOgBaseDataset):
    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]

        image_path = self.get_image_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")

        bbox = self.get_bbox(row["bbox"])

        return {"image": image, "bbox": bbox}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated_batch = {"images": [], "bboxes": []}

        for item in batch:
            collated_batch["images"].append(item["image"])
            collated_batch["bboxes"].append(torch.tensor(item["bbox"]))

        collated_batch["bboxes"] = torch.stack(collated_batch["bboxes"])  # type: ignore

        return collated_batch


if __name__ == "__main__":
    from pprint import pprint

    dataset = YOLOBaselineDataset()

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
