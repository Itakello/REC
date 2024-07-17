from dataclasses import dataclass, field

from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ..datasets.baseline_dataset import BaselineDataset
from ..interfaces.base_model import BaseModel
from ..models.clip_model import ClipModel
from ..utils.calculate_iou import calculate_iou


@dataclass
class BaselineModel(BaseModel):
    clip: ClipModel = field(default_factory=ClipModel)
    name: str = "baseline"
    iou_threshold: float = 0.8

    def __post_init__(self) -> None:
        super().__post_init__()

    def evaluate(self, dataloader: DataLoader) -> float:
        correct_predictions = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating baseline model"):
            batch_size = len(batch["candidates_embeddings"])
            total_samples += batch_size

            for i in range(batch_size):
                candidates_embeddings = batch["candidates_embeddings"][i]
                sentence_embedding = batch["sentence_embeddings"][i]
                gt_bbox = batch["gt_bboxes"][i]
                yolo_predictions = batch["yolo_predictions"][i]

                # Calculate similarities
                similarities = self.clip.get_similarity(
                    candidates_embeddings, sentence_embedding
                )
                most_similar_idx = similarities.argmax().item()

                # Get the most similar candidate's bbox
                most_similar_bbox = yolo_predictions[most_similar_idx]

                # Calculate IoU
                iou, _ = calculate_iou(most_similar_bbox, gt_bbox)

                # Check if the prediction is correct
                if iou >= self.iou_threshold:
                    correct_predictions += 1

        accuracy = 100 * correct_predictions / total_samples
        return accuracy

    def log_to_wandb(self, accuracy: float) -> None:
        wandb.log({"test/accuracy": accuracy}, step=4)
        wandb.log({"test/accuracy": accuracy}, step=9)

    def evaluate_and_log(self, test_dataset: BaselineDataset) -> None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
        )

        wandb.init(project="REC", name=f"{self.name}_evaluation")

        accuracy = self.evaluate(test_dataloader)
        self.log_to_wandb(accuracy)

        print(f"Evaluation Results:")
        print(f"IoU threshold {self.iou_threshold}: {accuracy:.4f}")

        wandb.finish()


if __name__ == "__main__":

    from ..utils.consts import CLIP_MODEL

    test_dataset = BaselineDataset(split="test")
    clip = ClipModel(version=CLIP_MODEL)
    baseline_model = BaselineModel(clip=clip)
    baseline_model.evaluate_and_log(test_dataset)
