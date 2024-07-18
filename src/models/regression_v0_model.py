from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


@dataclass
class RegressionV0Model(BaseCustomModel):
    name: str = "regression_v0"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.num_candidates * self.embeddings_dim * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 4)

    def forward(
        self,
        candidate_encodings: torch.Tensor,  # shape: [batch_size, num_candidates, embeddings_dim]
        sentence_encoding: torch.Tensor,  # shape: [batch_size, embeddings_dim]
        original_image_encoding: torch.Tensor,  # shape: [batch_size, embeddings_dim]
    ) -> torch.Tensor:
        # x shape: [batch_size, 6, 2048]
        combined_features = torch.cat(
            [
                original_image_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
                candidate_encodings,
                sentence_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
            ],
            dim=-1,
        )
        x = self.flatten(combined_features)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def __hash__(self) -> int:
        return super().__hash__()

    def prepare_input(self, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        candidates_embeddings = batch["candidates_embeddings"].to(DEVICE)
        combined_sentences_embeddings = batch["combined_sentences_embeddings"].to(
            DEVICE
        )
        images_embeddings = batch["image_features"].to(DEVICE)
        labels = batch["targets"].to(DEVICE)
        return [
            candidates_embeddings,
            combined_sentences_embeddings,
            images_embeddings,
        ], labels

    def calculate_accuracy(
        self, outputs: torch.Tensor, labels: torch.Tensor, iou_threshold: float = 0.8
    ) -> int:
        # Ensure the outputs and labels are on the same device
        outputs = outputs.to(labels.device)

        # Calculate IoUs for each bounding box
        ious = self.calculate_iou(outputs, labels)

        # Determine which predictions are correct based on the IoU threshold
        correct = ious >= iou_threshold

        # Calculate accuracy
        accuracy = correct.item()

        return accuracy


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
    }
    model = RegressionV0Model(config=config).to(DEVICE)
    print(f"Created new model with version number: {model.version_num}")

    # Test forward pass
    batch_size = 32
    candidate_encodings = torch.randn(batch_size, 6, 1024).to(DEVICE)
    sentence_encoding = torch.randn(batch_size, 1024).to(DEVICE)
    original_image_encoding = torch.randn(batch_size, 1024).to(DEVICE)

    # Forward pass
    output = model(candidate_encodings, sentence_encoding, original_image_encoding)

    print(
        f"Input shape: {candidate_encodings.shape}, {sentence_encoding.shape}, {original_image_encoding.shape}"
    )
    print(f"Output shape: {output.shape}")
