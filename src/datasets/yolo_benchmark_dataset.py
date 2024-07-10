from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from ..utils.consts import MODELS_PATH
from .refcocog_base_dataset import RefCOCOgBaseDataset


@dataclass
class YOLOBenchmarkDataset(RefCOCOgBaseDataset):
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

        collated_batch["bboxes"] = torch.stack(collated_batch["bboxes"])

        return collated_batch


if __name__ == "__main__":
    from pprint import pprint

    from ..classes.llm import LLM
    from ..managers.download_manager import DownloadManager
    from ..managers.preprocess_manager import PreprocessManager
    from ..models.clip_model import ClipModel
    from ..utils.consts import CLIP_MODEL, DATA_PATH, LLM_MODEL, LLM_SYSTEM_PROMPT_PATH

    dm = DownloadManager(data_path=DATA_PATH)

    llm = LLM(
        base_model=LLM_MODEL,
        system_prompt_path=LLM_SYSTEM_PROMPT_PATH,
    )
    clip = ClipModel(version=CLIP_MODEL, models_path=MODELS_PATH)
    pm = PreprocessManager(
        data_path=DATA_PATH,
        images_path=dm.images_path,
        raw_annotations_path=dm.annotations_path,
        llm=llm,
        clip=clip,
    )

    # Create a sample dataset
    dataset = RefCOCOgBaseDataset(
        annotations_path=pm.annotations_path,
        images_path=pm.images_path,
        embeddings_path=pm.embeddings_path,
        split="train",
        limit=10,
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
