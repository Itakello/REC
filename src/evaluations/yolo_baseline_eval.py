from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import seaborn as sns
from itakello_logging import ItakelloLogging
from tqdm import tqdm

import wandb

from ..classes.metric import Metrics
from ..datasets.yolo_baseline_dataset import YOLOBaselineDataset
from ..interfaces.base_eval import BaseEval
from ..models.yolo_model import YOLOModel
from ..utils.calculate_iou import calculate_iou
from ..utils.consts import STATS_PATH, WANDB_PROJECT
from ..utils.create_directory import create_directory

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class YOLOBaselineEval(BaseEval):
    iou_thresholds: list[float] = field(default_factory=list)
    yolo_versions: list[str] = field(default_factory=list)
    name: str = "yolo-baseline"
    models: dict[str, YOLOModel] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.models = {
            version: YOLOModel(version=version) for version in self.yolo_versions
        }
        create_directory(STATS_PATH)

    def evaluate(self) -> dict[str, Metrics]:
        results = {
            version: {iou: 0 for iou in self.iou_thresholds}
            for version in self.yolo_versions
        }
        total_samples = 0

        for batch in tqdm(
            YOLOBaselineDataset.get_dataloaders()["val"], desc="Evaluating YOLO models"
        ):
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
        heatmap_data = []
        for version, version_metrics in metrics.items():
            run = wandb.init(
                project=WANDB_PROJECT,  # type: ignore
                name=version,
            )
            wandb.log(
                {
                    f"{self.name}/{metric.name}": metric.value
                    for metric in version_metrics
                },
            )
            values = [metric.value for metric in version_metrics]
            heatmap_data.append(values)
            run.finish()

        x_labels = [f"{iou}" for iou in self.iou_thresholds]
        y_labels = list(metrics.keys())
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            data=heatmap_data,
            annot=True,
            fmt=".2f",
            xticklabels=x_labels,
            yticklabels=y_labels,
        )
        plt.title("YOLO Baseline Evaluation")
        plt.xlabel("IOU Threshold")
        plt.ylabel("YOLO Version")
        plt.savefig(STATS_PATH / f"{self.name}_heatmap.png")
        plt.close()


if __name__ == "__main__":
    from ..utils.consts import IOU_THRESHOLDS, YOLO_VERSIONS

    evaluator = YOLOBaselineEval(
        iou_thresholds=IOU_THRESHOLDS,
        yolo_versions=YOLO_VERSIONS[-2:],
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:")
    for version, version_metrics in metrics.items():
        print(f"{version}:")
        print(version_metrics)
