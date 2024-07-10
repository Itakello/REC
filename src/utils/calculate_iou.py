import torch


def calculate_iou(
    bbox_1: torch.Tensor, bbox_2: torch.Tensor, iou_threshold: float = 0.5
) -> tuple[float, bool]:
    xA = max(bbox_1[0], bbox_2[0])  # type: ignore
    yA = max(bbox_1[1], bbox_2[1])  # type: ignore
    xB = min(bbox_1[2], bbox_2[2])  # type: ignore
    yB = min(bbox_1[3], bbox_2[3])  # type: ignore

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1)
    box2Area = (bbox_2[2] - bbox_2[0] + 1) * (bbox_2[3] - bbox_2[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)
    iou_score = iou.item() if isinstance(iou, torch.Tensor) else iou
    iou_correct = iou_score > iou_threshold
    return iou_score, iou_correct
