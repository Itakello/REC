from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from ..classes.highlighting_modality import HighlightingModality


def save_image_with_bboxes(
    image: Image.Image,
    gt_bbox: list[float],
    pred_bboxes: list[list[float]] | torch.Tensor,
    path: Path,
) -> None:
    # Create a copy of the image to draw on
    draw_image = image.copy()

    # Draw ground truth bounding box in green
    draw_image = HighlightingModality.draw_rectangles(
        draw_image, [gt_bbox], color="green"
    )

    # Draw predicted bounding boxes in red
    draw_image = HighlightingModality.draw_rectangles(
        draw_image, pred_bboxes, color="red"
    )

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(draw_image)
    ax.axis("off")

    # Add legend
    ax.plot([], [], color="green", label="Ground Truth", linewidth=2)
    ax.plot([], [], color="red", label="Prediction", linewidth=2)
    ax.legend()

    # Save the figure
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Image saved to {path}")
