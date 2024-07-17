from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


def get_positional_encoding(seq_len, d_model) -> torch.Tensor:
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return torch.FloatTensor(pos_encoding)


@dataclass
class ClassificationV2Model(BaseCustomModel):
    name: str = "classification_v2"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()

        self.positional_encoding = get_positional_encoding(
            self.num_candidates, self.embeddings_dim
        )
        self.register_buffer("pe", self.positional_encoding)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.num_candidates * self.embeddings_dim * 2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, self.num_candidates + 1)

    def forward(
        self, candidates_embeddings: torch.Tensor, sentence_embedding: torch.Tensor
    ) -> torch.Tensor:

        # Apply positional encoding to candidates
        candidates_embeddings = candidates_embeddings + self.pe.unsqueeze(0)

        # Repeat sentence embedding for each candidate
        sentence_embeddings_repeated = sentence_embedding.repeat(
            1, self.num_candidates, 1
        )

        # Concatenate candidates and sentence embeddings
        combined = torch.cat(
            [candidates_embeddings, sentence_embeddings_repeated], dim=-1
        )

        x = self.flatten(combined)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return F.softmax(x, dim=1)

    def __hash__(self) -> int:
        return super().__hash__()

    def prepare_input(self, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        candidates_embeddings = batch["candidates_embeddings"]
        sentence_embedding = batch["combined_sentences_embeddings"]
        labels = batch["targets"]

        return [candidates_embeddings, sentence_embedding], labels

    def calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == torch.argmax(labels, dim=1)).sum().item()
        return int(correct)


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
    }
    model = ClassificationV2Model(config=config).to(DEVICE)
    print(f"Created new model with version number: {model.version_num}")

    # Test forward pass
    batch_size = 32
    candidates_embeddings = torch.randn(batch_size, 6, 1024).to(DEVICE)
    sentence_embedding = torch.randn(batch_size, 1024).to(DEVICE)

    # Forward pass
    output = model(candidates_embeddings, sentence_embedding)

    print(f"Candidates embeddings shape: {candidates_embeddings.shape}")
    print(f"Sentence embedding shape: {sentence_embedding.shape}")
    print(f"Output shape: {output.shape}")
