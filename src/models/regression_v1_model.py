from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE


@dataclass
class RegressionV0Model(BaseCustomModel):
    name: str = "regression-v0"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()

        # Define fully connected layers for combining features
        self.fc_comb1 = nn.Linear(self.embeddings_dim * 3, 512)
        self.bn_comb1 = nn.BatchNorm1d(512)
        self.fc_comb2 = nn.Linear(512, 256)
        self.bn_comb2 = nn.BatchNorm1d(256)
        self.fc_comb3 = nn.Linear(256, self.num_candidates)

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

        # Expand sentence encoding to match the dimensions of candidate encodings
        sentence_encoding_expanded = sentence_encoding.unsqueeze(1).repeat(
            1, self.num_candidates, 1
        )

        # Concatenate original image encoding, candidate encodings, and sentence encoding
        combined_features = torch.cat(
            [
                original_image_encoding.unsqueeze(1).repeat(1, self.num_candidates, 1),
                candidate_encodings,
                sentence_encoding_expanded,
            ],
            dim=-1,
        )

        # Process combined features to select the best candidate
        combined_features_flat = combined_features.view(
            batch_size * self.num_candidates, -1
        )
        x_comb = F.relu(self.bn_comb1(self.fc_comb1(combined_features_flat)))
        x_comb = F.relu(self.bn_comb2(self.fc_comb2(x_comb)))
        candidate_scores = self.fc_comb3(x_comb).view(batch_size, self.num_candidates)

        # Select the best candidate based on the scores
        best_candidate_indices = candidate_scores.argmax(dim=1)

        # Gather the selected candidate encodings and bounding boxes
        selected_candidate_embeddings = candidate_encodings[
            range(batch_size), best_candidate_indices
        ]
        selected_bboxes = bounding_boxes[range(batch_size), best_candidate_indices]

        # Concatenate the original image encoding with the selected candidate encoding
        final_features = torch.cat(
            [original_image_encoding, selected_candidate_embeddings], dim=-1
        )

        # Pass through fully connected layers to regress bounding box coordinates
        x = F.relu(self.bn1(self.fc1(final_features)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        coordinates = self.fc4(x)  # Shape: [batch_size, 4]

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

    def calculate_accuracy(self, outputs, labels, iou_threshold=0.8):
        # Ensure the outputs and labels are on the same device
        outputs = outputs.to(labels.device)

        # Calculate IoUs for each bounding box
        ious = self.calculate_iou(outputs, labels)

        # Determine which predictions are correct based on the IoU threshold
        correct = ious >= iou_threshold

        # Calculate accuracy
        accuracy = correct.float().mean().item()

        return accuracy
