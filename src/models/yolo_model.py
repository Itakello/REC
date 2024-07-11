from dataclasses import dataclass, field
from pathlib import Path

import torch
from itakello_logging import ItakelloLogging
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from ..datasets.yolo_baseline_dataset import YOLOBaselineDataset
from ..interfaces.base_model import BaseModel
from ..utils.consts import DATA_PATH, DEVICE, MODELS_PATH
from ..utils.create_directory import create_directory
from ..utils.save_sample import save_and_visualize_image_with_bboxes

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class YOLOModel(BaseModel):
    name: str = field(init=False, default="yolo")
    model: YOLO = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model = YOLO(self.model_path / f"{self.version}.pt").to(DEVICE)

    def get_bboxes(self, image: Image.Image) -> torch.Tensor:
        predictions = self.model.predict(source=image, verbose=False)[0]  # type: ignore

        assert isinstance(predictions.boxes, Boxes)
        if len(predictions.boxes) > 0:
            return predictions.boxes.xyxy.to(DEVICE)  # type: ignore
        else:
            return (
                torch.Tensor([0, 0, image.width, image.height]).unsqueeze(0).to(DEVICE)
            )


if __name__ == "__main__":
    # Initialize the YOLO model with the "yolov5mu" version
    yolo_model = YOLOModel(version="yolov5mu", models_path=MODELS_PATH)

    # Initialize the YOLOBenchmarkDataset
    dataset = YOLOBaselineDataset(
        annotations_path=DATA_PATH / "annotations.csv",
        images_path=DATA_PATH / "images",
        embeddings_path=DATA_PATH / "embeddings",
        split="train",
        limit=1,
    )

    # Get the first sample from the dataset
    first_sample = dataset[0]
    image = first_sample["image"]
    gt_bbox = first_sample["bbox"]

    # Get bounding box predictions from the YOLO model
    predicted_bboxes = yolo_model.get_bboxes(image)

    # Log the results
    logger.info(f"Ground truth bbox: {gt_bbox}")
    logger.info(f"Predicted bboxes: {predicted_bboxes}")

    samples_dir = create_directory(Path("./samples"))

    # Save and visualize the results
    save_and_visualize_image_with_bboxes(
        image, gt_bbox, predicted_bboxes, samples_dir / "yolo_prediction_result.jpg"
    )
