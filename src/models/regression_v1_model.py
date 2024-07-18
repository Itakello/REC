from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.calculate_iou import calculate_iou
from ..utils.consts import DEVICE


@dataclass
class RegressionV1Model(BaseCustomModel):
    name: str = "regression_v1"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self.embeddings_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 4)

    def forward(
        self,
        sentence_encoding: torch.Tensor,  # [32,1024]
        original_image_encoding: torch.Tensor,  # [32,1024]
        bounding_boxes: torch.Tensor,  # [32,6,4]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        top_bounding_box = bounding_boxes[:, 0, :]  # shape: [batch_size, 4]

        regression_inputs = torch.cat(
            [original_image_encoding, sentence_encoding],
            dim=-1,
        )
        x = F.relu(self.bn1(self.fc1(regression_inputs)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x + top_bounding_box

    def __hash__(self) -> int:
        return super().__hash__()

    def prepare_input(self, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        candidate_encodings = batch["candidates_embeddings"].to(DEVICE)
        sentence_encoding = batch["combined_sentences_embeddings"].to(DEVICE).squeeze()
        original_image_encoding = batch["image_embeddings"].to(DEVICE).squeeze()
        bounding_boxes = batch["candidates_bboxes"].to(DEVICE)
        gold_embeddings = batch["gold_embeddings"].to(DEVICE)
        labels = batch["gold_bboxes"].to(DEVICE)
        images = batch["images"]

        return [
            candidate_encodings,
            sentence_encoding,
            original_image_encoding,
            bounding_boxes,
            gold_embeddings,
            images,
        ], labels

    def calculate_accuracy(self, outputs, labels, iou_threshold: float = 0.8) -> float:
        # Ensure the outputs and labels are on the same device
        outputs = outputs.to(labels.device)

        corrects = 0
        for output, label in zip(outputs, labels):
            _, correct = calculate_iou(output, label, iou_threshold)
            if correct:
                corrects += 1

        # Determine which predictions are correct based on the IoU threshold

        # Calculate accuracy
        accuracy = corrects / len(outputs)

        return accuracy


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
    }
    model = RegressionV1Model(config=config).to(DEVICE)
    print(f"Created new model with version number: {model.version_num}")

    # Test forward pass
    batch_size = 32
    candidate_encodings = torch.randn(batch_size, 6, 1024).to(DEVICE)
    sentence_encoding = torch.randn(batch_size, 1024).to(DEVICE)
    original_image_encoding = torch.randn(batch_size, 1024).to(DEVICE)
    bounding_boxes = torch.randn(batch_size, 6, 4).to(DEVICE)

    # Forward pass
    output = model(
        candidate_encodings, sentence_encoding, original_image_encoding, bounding_boxes
    )

    print(
        f"Input shape: {candidate_encodings.shape}, {sentence_encoding.shape}, {original_image_encoding.shape}"
    )
    print(f"Output shape: {output.shape}")
