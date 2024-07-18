import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops.boxes import generalized_box_iou

from ..models.clip_model import ClipModel
from ..utils.consts import CLIP_MODEL, DEVICE
from .highlighting_modality import HighlightingModality


class SemanticLoss(nn.Module):
    def __init__(self) -> None:
        super(SemanticLoss, self).__init__()
        self.clip = ClipModel(version=CLIP_MODEL)

    def forward(
        self,
        img_embed_pred: torch.Tensor,
        img_embed_gt: torch.Tensor,
        caption_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the semantic loss using precomputed CLIP embeddings.
        img_embed_pred: CLIP embeddings of the region in the predicted bounding box.
        img_embed_gt: CLIP embeddings of the region in the gold bounding box.
        caption_embed: CLIP embeddings of the caption.
        """
        # Cosine similarity between caption embeddings and predicted/gold image embeddings
        cosine_sim_pred = self.clip.get_similarity(
            img_embed_pred.squeeze(), caption_embed
        )
        cosine_sim_gt = self.clip.get_similarity(img_embed_gt.squeeze(), caption_embed)

        # Semantic loss based on the difference in similarities
        semantic_loss = torch.mean(torch.abs(cosine_sim_pred - cosine_sim_gt))

        return semantic_loss


class RECLoss(nn.Module):
    def __init__(
        self, lambda_spatial: float = 1.0, lambda_semantic: float = 1.0
    ) -> None:
        super(RECLoss, self).__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_semantic = lambda_semantic
        self.semantic_loss = SemanticLoss()

    def generalized_iou_loss(
        self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        giou = generalized_box_iou(pred_boxes, gt_boxes)
        giou_loss = 1 - giou  # Loss is 1 - IoU
        return giou_loss.mean()

    def compute_pred_images(
        self, images: list[Image.Image], pred_bboxes: torch.Tensor
    ) -> torch.Tensor:
        pred_images = []
        for image, pred_bbox in zip(images, pred_bboxes):
            highlighted_image = HighlightingModality.apply_highlighting(
                image, pred_bbox, "crop"
            )
            embedding_image = self.semantic_loss.clip.encode_images(highlighted_image)
            pred_images.append(embedding_image)
        return torch.stack(pred_images).to(DEVICE)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_images: torch.Tensor,
        captions: torch.Tensor,
        images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_loss = self.generalized_iou_loss(pred_boxes, gt_boxes)
        # TODO calculate pred_images
        pred_images = self.compute_pred_images(images, pred_boxes)
        semantic_loss = self.semantic_loss(pred_images, gt_images, captions)

        total_loss = (
            self.lambda_spatial * spatial_loss + self.lambda_semantic * semantic_loss
        )
        return total_loss, spatial_loss, semantic_loss
