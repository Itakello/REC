from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


@dataclass
class ClassificationV3Model(BaseCustomModel):
    name: str = "classification_v3"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)
    num_heads: int = field(default=8)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()

        self.positional_encoding = nn.Parameter(
            self.get_positional_encoding(self.num_candidates, self.embeddings_dim)
        )

        self.attention = CrossModalAttention(self.embeddings_dim, self.num_heads)

        self.fc1 = nn.Linear(self.embeddings_dim * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)

    @staticmethod
    def get_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(
        self, candidates_embeddings: torch.Tensor, sentence_embedding: torch.Tensor
    ) -> torch.Tensor:
        batch_size = candidates_embeddings.size(0)

        # Apply positional encoding to candidates
        candidates_embeddings = (
            candidates_embeddings + self.positional_encoding.unsqueeze(0)
        )

        # Reshape for attention: (seq_len, batch_size, embed_dim)
        candidates_embeddings = candidates_embeddings.transpose(0, 1)
        sentence_embedding = sentence_embedding.unsqueeze(0).repeat(
            self.num_candidates, 1, 1
        )

        # Apply cross-modal attention
        attended_features = self.attention(
            candidates_embeddings, sentence_embedding, candidates_embeddings
        )

        # Reshape back: (batch_size, seq_len, embed_dim)
        attended_features = attended_features.transpose(0, 1)

        # Concatenate attended features with sentence embedding for each candidate
        sentence_embedding_expanded = sentence_embedding.transpose(0, 1)
        combined = torch.cat([attended_features, sentence_embedding_expanded], dim=-1)

        # Process each candidate separately
        outputs = []
        for i in range(self.num_candidates):
            x = combined[:, i, :]
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            outputs.append(x)

        # Stack outputs and add the "no correct candidate" option
        outputs = torch.cat(outputs, dim=1)
        no_candidate = torch.zeros(batch_size, 1).to(outputs.device)
        outputs = torch.cat([outputs, no_candidate], dim=1)

        return F.softmax(outputs, dim=1)

    def __hash__(self) -> int:
        return super().__hash__()

    def prepare_input(self, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        candidates_embeddings = batch["candidates_embeddings"].to(DEVICE)
        sentence_embedding = batch["combined_sentences_embeddings"].to(DEVICE)
        labels = batch["targets"].to(DEVICE)

        return [candidates_embeddings, sentence_embedding], labels

    def calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> int:
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == torch.argmax(labels, dim=1)).sum().item()
        return int(correct)


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
        "num_heads": 8,
    }
    model = ClassificationV3Model(config=config).to(DEVICE)
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
