from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


@dataclass
class ClassificationV0Model(BaseCustomModel):
    name: str = "classification_v0"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.num_candidates * self.embeddings_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, self.num_candidates + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 6, 2048]
        x = self.flatten(x)  # shape: [batch_size, 6 * 2048]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # shape: [batch_size, 6]
        return x

    def __hash__(self) -> int:
        return super().__hash__()

    def prepare_input(self, batch):
        inputs = {
            "candidates_embeddings": batch["candidates_embeddings"].to(DEVICE),
            "combined_sentences_embeddings": batch["combined_sentences_embeddings"].to(
                DEVICE
            ),
        }
        labels = batch["targets"].to(DEVICE)
        return inputs, labels

    def calculate_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == torch.argmax(labels, dim=1)).sum().item()
        return correct


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
    }
    model = ClassificationV0Model(config=config).to(DEVICE)
    print(f"Created new model with version number: {model.version_num}")

    # Test forward pass
    batch_size = 32
    sample_input = torch.randn(batch_size, 6, 2048).to(DEVICE)

    # Forward pass
    output = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
