from dataclasses import dataclass, field

from itakello_logging import ItakelloLogging
from torch.utils.data import DataLoader

import wandb

from ..classes.metric import Metrics
from ..datasets.yolo_baseline_dataset import YOLOBaselineDataset
from ..interfaces.base_eval import BaseEval
from ..models.yolo_model import YOLOModel
from ..utils.calculate_iou import calculate_iou
from ..utils.consts import DATA_PATH, MODELS_PATH, WANDB_PROJECT

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class YOLOBaselineEval(BaseEval):
    iou_thresholds: list[float] = field(default_factory=list)
    yolo_versions: list[str] = field(default_factory=list)
    name: str = "yolo-baseline"
    dataset: YOLOBaselineDataset = field(init=False)
    models: dict[str, YOLOModel] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dataset = YOLOBaselineDataset(
            annotations_path=DATA_PATH / "annotations.csv",
            images_path=DATA_PATH / "images",
            embeddings_path=DATA_PATH / "embeddings",
            split="train",
        )
        self.models = {
            version: YOLOModel(version=version, models_path=MODELS_PATH)
            for version in self.yolo_versions
        }

    def get_dataloaders(self) -> list[tuple[str, DataLoader]]:
        return [
            (
                "train",
                DataLoader(
                    self.dataset,
                    collate_fn=self.dataset.collate_fn,
                ),
            )
        ]

    def evaluate(self) -> dict[str, Metrics]:
        results = {
            version: {iou: 0 for iou in self.iou_thresholds}
            for version in self.yolo_versions
        }
        total_samples = 0

        for batch in self.get_dataloaders()[0][1]:
            images = batch["images"]
            gt_bboxes = batch["bboxes"]

            for image, gt_bbox in zip(images, gt_bboxes):
                for version, model in self.models.items():
                    predictions = model.get_bboxes(image)

                    best_iou = 0
                    for pred_bbox in predictions:
                        iou = calculate_iou(pred_bbox, gt_bbox)[0]
                        best_iou = max(best_iou, iou)

                    for threshold in self.iou_thresholds:
                        if best_iou >= threshold:
                            results[version][threshold] += 1

                total_samples += 1

        # Normalize results and convert to Metrics objects
        metrics = {}
        for version in results:
            metrics[version] = Metrics()
            for iou, value in results[version].items():
                metrics[version].add(f"iou_{iou}", value / total_samples)

        self.log_metrics(metrics)
        logger.confirmation("Yolo evaluation completed")
        return metrics

    def log_metrics(self, metrics: Metrics | dict[str, Metrics]) -> None:
        assert isinstance(metrics, dict)
        for version, version_metrics in metrics.items():
            run = wandb.init(
                project=WANDB_PROJECT,
                name=version,
            )
            wandb.log(
                {
                    f"{self.name}/{metric.name}": metric.value
                    for metric in version_metrics
                },
            )
            run.finish()


if __name__ == "__main__":
    from ..utils.consts import IOU_THRESHOLDS, YOLO_VERSIONS

    evaluator = YOLOBaselineEval(
        iou_thresholds=IOU_THRESHOLDS,
        yolo_versions=YOLO_VERSIONS,
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:")
    for version, version_metrics in metrics.items():
        print(f"{version}:")
        print(version_metrics)
