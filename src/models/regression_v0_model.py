from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(
            embed_dim, embed_dim * 2
        )  # project sentence to match concatenated encoding
        self.key_proj = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.value_proj = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(key.shape[-1], dtype=torch.float32)
        )
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


@dataclass
class RegressionModel(BaseCustomModel):
    name: str = "ranking"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()
        self.attention = CrossModalAttention(self.embeddings_dim)

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
    ):

        batch_size = candidate_encodings.size(0)

        # Concatenate original image encoding with each candidate encoding
        concatenated_encodings = torch.cat(
            [
                original_image_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
                candidate_encodings,
            ],
            dim=-1,
        )

        # Apply attention between concatenated encodings and sentence encoding
        sentence_encoding = sentence_encoding.unsqueeze(1).repeat(
            1, self.num_candidates, 1
        )
        attended_encodings, attention_weights = self.attention(
            sentence_encoding, concatenated_encodings, concatenated_encodings
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

    def calculate_iou(boxA, boxB):
        # Compute the (x, y)-coordinates of the intersection rectangle
        xA = torch.max(boxA[:, 0], boxB[:, 0])
        yA = torch.max(boxA[:, 1], boxB[:, 1])
        xB = torch.min(boxA[:, 2], boxB[:, 2])
        yB = torch.min(boxA[:, 3], boxB[:, 3])

        # Compute the area of intersection rectangle
        interArea = torch.max(xB - xA, torch.zeros_like(xB)) * torch.max(
            yB - yA, torch.zeros_like(yB)
        )

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
        boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

        # Compute the intersection over union
        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    def calculate_accuracy(self, outputs, labels, iou_threshold=0.8):
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
        ious = self.calculate_iou(outputs, labels)

        # Determine which predictions are correct based on the IoU threshold
        correct = ious >= iou_threshold

        # Calculate accuracy
        accuracy = correct.float().mean().item()

        return accuracy
