from dataclasses import dataclass

from PIL import Image

from .refcocog_base_dataset import RefCOCOgBaseDataset


@dataclass
class YOLOBenchmarkDataset(RefCOCOgBaseDataset):
    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]

        image_path = self.get_image_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")

        bbox = self.get_bbox(row["bbox"])

        return {"image": image, "bbox": bbox}
