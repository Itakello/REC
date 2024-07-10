import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from itakello_logging import ItakelloLogging
from PIL import Image

from ..interfaces.base_dataset import BaseDataset
from ..utils.consts import CLIP_MODEL, MODELS_PATH

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class RefCOCOgBaseDataset(BaseDataset):
    annotations_path: Path
    images_path: Path
    embeddings_path: Path
    split: str = "train"
    limit: int = -1
    data: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._load_data()
        self._filter_data()

    def _load_data(self) -> None:
        self.data = pd.read_csv(self.annotations_path)
        logger.info(f"Loaded {len(self.data)} total samples")

    def _filter_data(self) -> None:
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
        return self.images_path / file_name

    def get_bbox(self, bbox_str: str) -> list:
        return json.loads(bbox_str)

    def get_embeddings(
        self, embeddings_filename: str, keys: list[str] | None = None
    ) -> dict:
        full_path = self.embeddings_path / embeddings_filename
        embeddings = np.load(full_path)

        if keys is None:
            return {k: embeddings[k] for k in embeddings.keys()}

        result = {}
        for key in keys:
            if key not in embeddings:
                raise KeyError(f"Embedding key '{key}' not found in {full_path}")
            result[key] = embeddings[key]

        return result

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]

        image_path = self.get_image_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")

        bbox = self.get_bbox(row["bbox"])

        embeddings = self.get_embeddings(row["embeddings_filename"])

        return {
            "image": image,
            "bbox": bbox,
            "file_name": row["file_name"],
            "embeddings": embeddings,
            "sentences": row["sentences"],
            "comprehensive_sentence": row["comprehensive_sentence"],
            "category": row["category"],
            "supercategory": row["supercategory"],
            "area": row["area"],
        }


if __name__ == "__main__":

    from pprint import pprint

    from ..classes.llm import LLM
    from ..managers.download_manager import DownloadManager
    from ..managers.preprocess_manager import PreprocessManager
    from ..models.clip_model import ClipModel
    from ..utils.consts import DATA_PATH, LLM_MODEL, LLM_SYSTEM_PROMPT_PATH

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
    pprint(printable_item, depth=2, compact=True)
