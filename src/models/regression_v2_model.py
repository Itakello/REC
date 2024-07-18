from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.calculate_iou import calculate_iou
from ..utils.consts import DEVICE


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(self, query, key, value) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_weights


@dataclass
class RegressionV0Model(BaseCustomModel):
    name: str = "regression_v0"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)
    num_heads: int = field(default=8)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()
        self.attention = CrossModalAttention(self.embeddings_dim * 2, self.num_heads)

        # Define fully connected layers for regression
        self.fc1 = nn.Linear(self.embeddings_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 4)  # Regressing bounding box coordinates (x, y, w, h)

    def forward(
        self,
        candidate_encodings,
        sentence_encoding,
        original_image_encoding,
        bounding_boxes,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = candidate_encodings.size(0)

        # Concatenate original image encoding with each candidate encoding
        concatenated_candidates_encodings = torch.cat(
            [
                original_image_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
                candidate_encodings,
            ],
            dim=-1,
        )

        concatenated_sentences_encodings = torch.cat(
            [
                original_image_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
                sentence_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
            ],
            dim=-1,
        )

        attended_encodings, attention_weights = self.attention(
            concatenated_sentences_encodings,
            concatenated_candidates_encodings,
            concatenated_candidates_encodings,
        )

        # Find the index of the candidate with the highest attention weight
        max_attention_indices = attention_weights.argmax(dim=1)

        # Select the bounding box corresponding to the highest attention weight for each item in the batch
        selected_bboxes = bounding_boxes[range(batch_size), max_attention_indices]
        # Select the candidate embedding corresponding to the highest attention weight for each item in the batch
        selected_candidate_embeddings = candidate_encodings[
            range(batch_size), max_attention_indices
        ]

        # Flatten the attended encodings
        attended_encodings = self.flatten(attended_encodings)

        # Pass through fully connected layers to regress bounding box coordinates
        x = F.relu(self.bn1(self.fc1(attended_encodings)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        coordinates = self.fc4(x)  # Shape: [batch_size, num_candidates, 4]

        # Add the regression output to the selected bounding boxes to get the final coordinates
        final_coordinates = selected_bboxes + coordinates

        return final_coordinates, selected_candidate_embeddings, sentence_encoding

    def __hash__(self) -> int:
        return super().__hash__()

    def prepare_input(self, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        candidate_encodings = batch["candidates_embeddings"].to(DEVICE)
        sentence_encoding = batch["combined_sentences_embeddings"].to(DEVICE)
        original_image_encoding = batch["image_embeddings"].to(DEVICE)
        bounding_boxes = batch["candidates_bboxes"].to(DEVICE)
        labels = batch["candidates_bboxes"].to(DEVICE)

        return [
            candidate_encodings,
            sentence_encoding,
            original_image_encoding,
            bounding_boxes,
        ], labels

    def calculate_accuracy(self, outputs, labels, iou_threshold=0.8):  # -> Any:
        """
        Calculates the accuracy of bounding box predictions.
        Args:
            outputs (torch.Tensor): Predicted bounding boxes of shape (batch_size, num_candidates, 4).
            labels (torch.Tensor): Ground truth bounding boxes of shape (batch_size, num_candidates, 4).
            iou_threshold (float): IoU threshold to consider a prediction as correct.
        Returns:
            float: Accuracy of the predictions.
        """
        # Ensure the outputs and labels are on the same device
        outputs = outputs.to(labels.device)

        # Calculate IoUs for each bounding box
        ious = calculate_iou(outputs, labels)

        # Determine which predictions are correct based on the IoU threshold
        correct = ious >= iou_threshold

        # Calculate accuracy
        accuracy = correct.float().mean().item()

        return accuracy


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
        "num_heads": 8,
    }
    model = RegressionV0Model(config=config).to(DEVICE)
    print(f"Created new model with version number: {model.version_num}")

    # Test forward pass
    batch_size = 32
    candidates_embeddings = torch.randn(batch_size, 6, 1024).to(DEVICE)
    sentence_embedding = torch.randn(batch_size, 1024).to(DEVICE)
    image_embedding = torch.randn(batch_size, 1024).to(DEVICE)
    bounding_boxes = torch.randn(batch_size, 6, 4).to(DEVICE)

    # Forward pass
    output = model(
        candidates_embeddings, sentence_embedding, image_embedding, bounding_boxes
    )

    print(f"Candidates embeddings shape: {candidates_embeddings.shape}")
    print(f"Sentence embedding shape: {sentence_embedding.shape}")
    print(f"Output shape: {output.shape}")
