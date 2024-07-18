from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F

from ..interfaces.base_custom_model import BaseCustomModel
from ..models.classification_v0_model import ClassificationV0Model
from ..utils.consts import DEVICE


@dataclass
class RegressionV2Model(BaseCustomModel):
    name: str = "regression_v2"
    embeddings_dim: int = field(default=1024)
    num_candidates: int = field(default=6)
    attention_heads: int = field(default=4)
    img_width: int = field(default=1920)  # test size
    img_height: int = field(default=1080)  # test size

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.flatten = nn.Flatten()

        self.attention = MultiheadAttention(
            embed_dim=self.embeddings_dim * 2, num_heads=self.attention_heads
        )
        self.attention_fc = nn.Linear(
            self.embeddings_dim * 2, 4
        )  # Projecting attention output to bounding box

        self.fc1 = nn.Linear(self.embeddings_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 4)

    def forward(
        self,
        candidate_encodings,
        sentence_encoding,
        original_image_encoding,
        bounding_boxes,
    ):

        # here we can do classification using one of our classification models or with attention
        # or simply rely on the baseline ranking: [candidate_encodings[:, 0, :]

        combined_encodings = torch.cat(
            [original_image_encoding, sentence_encoding], dim=-1
        ).unsqueeze(0)
        combined_encodings = combined_encodings.permute(
            1, 0, 2
        )  # (batch_size, 1, embeddings_dim * 2) -> (1, batch_size, embeddings_dim * 2)

        # Apply self-attention mechanism
        attention_output, _ = self.attention(
            combined_encodings, combined_encodings, combined_encodings
        )
        attention_output = attention_output.permute(1, 0, 2).squeeze(
            1
        )  # (1, batch_size, embeddings_dim * 2) -> (batch_size, embeddings_dim * 2)

        # Project the attention output to bounding box coordinates
        attention_output = self.attention_fc(attention_output)

        # Ensure the attention output is within valid bounding box coordinates
        attention_output = self.scale_to_range(attention_output)

        regression_inputs = torch.cat(
            [candidate_encodings[:, 0, :], original_image_encoding], dim=-1
        )
        x = F.relu(self.bn1(self.fc1(regression_inputs)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x + bounding_boxes[:, 0, :] + attention_output

    def __hash__(self) -> int:
        return super().__hash__()

    def scale_to_range(self, x) -> torch.Tensor:
        # Flatten the output and apply sigmoid
        x = torch.sigmoid(x)
        # Scale sigmoid output to image dimensions
        x[:, 0] *= self.img_width  # x
        x[:, 1] *= self.img_height  # y
        x[:, 2] *= self.img_height  # h
        x[:, 3] *= self.img_width  # w

        # Ensure valid bounding box coordinates
        x[:, 0] = torch.clamp(x[:, 0], min=0, max=self.img_width)  # x
        x[:, 1] = torch.clamp(x[:, 1], min=0, max=self.img_height)  # y
        max_height = self.img_height - x[:, 1]
        max_width = self.img_width - x[:, 0]
        x[:, 2] = torch.min(
            torch.max(x[:, 2], torch.tensor(1.0).to(x.device)), max_height
        )  # h
        x[:, 3] = torch.min(
            torch.max(x[:, 3], torch.tensor(1.0).to(x.device)), max_width
        )  # w

        return x

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

    ### check from here.     def calculate_iou(self, boxes1, boxes2):
    def calculate_iou(self, boxes1, boxes2):
        # Convert (x, y, h, w) to (x1, y1, x2, y2)
        boxes1 = self.convert_to_corners(boxes1)
        boxes2 = self.convert_to_corners(boxes2)

        # Calculate intersection
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection

        return intersection / union

    def convert_to_corners(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 3]  # x + w
        y2 = boxes[:, 1] + boxes[:, 2]  # y + h
        return torch.stack([x1, y1, x2, y2], dim=1)

    def bounding_box_loss(self, pred_boxes, target_boxes):
        # Standard loss (e.g., Smooth L1)
        loss = nn.SmoothL1Loss()(pred_boxes, target_boxes)
        # Adding constraint to ensure valid width and height
        width = pred_boxes[:, 3]
        height = pred_boxes[:, 2]
        x = pred_boxes[:, 0]
        y = pred_boxes[:, 1]

        constraint_loss = (
            torch.mean(torch.relu(-width))  # width should be positive
            + torch.mean(torch.relu(-height))  # height should be positive
            + torch.mean(
                torch.relu(x + width - self.img_width)
            )  # x + w should be within image width
            + torch.mean(
                torch.relu(y + height - self.img_height)
            )  # y + h should be within image height
        )

        total_loss = loss + constraint_loss
        return total_loss


if __name__ == "__main__":
    # Test the model
    config = {
        "learning_rate": 0.001,
    }
    model = RegressionV2Model().to(DEVICE)
    print(f"Created new model with version number: {model.name}")

    # Test forward pass
    batch_size = 32
    embeddings_dim = 1024
    num_candidates = 6

    # Create dummy data
    candidate_encodings = torch.randn(batch_size, num_candidates, embeddings_dim).to(
        DEVICE
    )
    sentence_encoding = torch.randn(batch_size, embeddings_dim).to(DEVICE)
    original_image_encoding = torch.randn(batch_size, embeddings_dim).to(DEVICE)
    bounding_boxes = torch.randn(batch_size, num_candidates, 4).to(DEVICE)

    # Forward pass
    output = model(
        candidate_encodings, sentence_encoding, original_image_encoding, bounding_boxes
    )

    print(
        f"Input shapes: candidate_encodings: {candidate_encodings.shape}, sentence_encoding: {sentence_encoding.shape}, original_image_encoding: {original_image_encoding.shape}, bounding_boxes: {bounding_boxes.shape}"
    )
    print(f"Output shape: {output.shape}")
