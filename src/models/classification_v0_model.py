from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


@dataclass
class ClassificationV0Model(BaseCustomModel):
    name: str = "classification_v0"
    input_dim: int = field(default=1024 * 2)  # 1024 for image, 1024 for text
    num_candidates: int = field(default=6)
    num_layers: int = field(default=3)
    neurons_per_layer: list[int] = field(default_factory=lambda: [256, 128, 64])

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()

        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.neurons_per_layer[i]))
            else:
                layers.append(
                    nn.Linear(self.neurons_per_layer[i - 1], self.neurons_per_layer[i])
                )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.neurons_per_layer[i]))

        layers.append(nn.Linear(self.neurons_per_layer[-1], 1))

        return nn.Sequential(*layers)

    def forward(
        self, candidates_embeddings: torch.Tensor, sentence_embedding: torch.Tensor
    ) -> torch.Tensor:
        batch_size = candidates_embeddings.size(0)

        # Repeat sentence embedding for each candidate
        sentence_embedding = sentence_embedding.unsqueeze(1).repeat(
            1, self.num_candidates, 1
        )

        # Concatenate candidate embeddings with sentence embedding
        combined = torch.cat([candidates_embeddings, sentence_embedding], dim=-1)

        # Reshape for processing
        combined = combined.view(batch_size * self.num_candidates, -1)

        # Pass through the network
        logits = self.network(combined)

        # Reshape back to (batch_size, num_candidates)
        logits = logits.view(batch_size, self.num_candidates)

        return logits

    def __hash__(self) -> int:
        return super().__hash__()


if __name__ == "__main__":
    # Test the model
    config = {
        "input_dim": 2048,
        "num_candidates": 6,
        "num_layers": 3,
        "neurons_per_layer": [256, 128, 64],
        "learning_rate": 0.001,
    }
    model = ClassificationV0Model(config=config).to(DEVICE)
    print(f"Created new model with version number: {model.version_num}")

    # Test forward pass
    batch_size = 32
    candidates_embeddings = torch.randn(batch_size, 6, 1024).to(DEVICE)
    sentence_embedding = torch.randn(batch_size, 1024).to(DEVICE)

    output = model(candidates_embeddings, sentence_embedding)
    print(f"Output shape: {output.shape}")

    # Test saving and loading
    model.save_checkpoint(
        epoch=1, optimizer=torch.optim.Adam(model.parameters()), loss=0.5
    )
    loaded_model = ClassificationV0Model.load_from_config(model.name, model.version_num)
    print(f"Loaded model configuration: {loaded_model.config}")

    print("Model test completed successfully!")
