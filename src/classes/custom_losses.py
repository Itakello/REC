import torch
import torch.nn as nn
from torchvision.ops.boxes import generalized_box_iou

from ..models.clip_model import ClipModel
from ..utils.consts import CLIP_MODEL


class SemanticLoss(nn.Module):
    def __init__(self) -> None:
        super(SemanticLoss, self).__init__()
        self.clip = ClipModel(version=CLIP_MODEL)

    def forward(self, img_embed_pred, img_embed_gt, caption_embed) -> torch.Tensor:
        """
        Compute the semantic loss using precomputed CLIP embeddings.
        img_embed_pred: CLIP embeddings of the region in the predicted bounding box.
        img_embed_gt: CLIP embeddings of the region in the gold bounding box.
        caption_embed: CLIP embeddings of the caption.
        """
        # Cosine similarity between caption embeddings and predicted/gold image embeddings
        cosine_sim_pred = self.clip.get_similarity(caption_embed, img_embed_pred)
        cosine_sim_gt = self.clip.get_similarity(caption_embed, img_embed_gt)

        # Semantic loss based on the difference in similarities
        semantic_loss = torch.mean(torch.abs(cosine_sim_pred - cosine_sim_gt))

        return semantic_loss


class RECLoss(nn.Module):
    def __init__(self, lambda_spatial=1.0, lambda_semantic=1.0) -> None:
        super(RECLoss, self).__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_semantic = lambda_semantic
        self.semantic_loss = SemanticLoss()

    def generalized_iou_loss(self, pred_boxes, gt_boxes):
        """
        Compute the Generalized IoU loss between predicted and ground truth boxes.
        """
        giou = generalized_box_iou(pred_boxes, gt_boxes)
        giou_loss = 1 - giou  # Loss is 1 - IoU
        return giou_loss.mean()

    def forward(self, images, pred_images, pred_boxes, gt_boxes, captions):
        spatial_loss = self.generalized_iou_loss(pred_boxes, gt_boxes)
        semantic_loss = self.semantic_loss(pred_images, images, captions)

        total_loss = (
            self.lambda_spatial * spatial_loss + self.lambda_semantic * semantic_loss
        )
        return total_loss, spatial_loss, semantic_loss
