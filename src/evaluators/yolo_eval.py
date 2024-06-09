from dataclasses import dataclass

from ..interfaces.base_eval import BaseEvaluator

# import wandb
# from torch.utils.data import DataLoader


@dataclass
class YOLOEvaluator(BaseEvaluator):
    yolo_versions: list[str]
    iou_thresholds: list[float]

    def evaluate(self) -> None:
        for yolo_version in self.yolo_versions:
            # _, val_loader, _ = get_dataloaders()
            """wandb.init(
                project="REC",
                name=yolo_version,
                config={"samples": val_loader.dataset.__sizeof__},
            )
            for iou_threshold in self.iou_thresholds:
                wandb.log({f"{self.eval_name}/iou_{iou_threshold}": 0})"""
            pass

    def get_dataloaders(self) -> list[tuple]:
        return super().get_dataloaders()
