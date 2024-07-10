import matplotlib.pyplot as plt
import torch
from PIL import Image

from .selection_modality import SelectionModality


def save_and_visualize_image_with_bboxes(
    image: Image.Image,
    gt_bbox: list[float],
    pred_bboxes: list[list[float]] | torch.Tensor,
    file_name: str = "output_image.png",
) -> None:
    # Create a copy of the image to draw on
    draw_image = image.copy()

    # Draw ground truth bounding box in green
    draw_image = SelectionModality.draw_rectangles(draw_image, [gt_bbox], color="green")

    # Draw predicted bounding boxes in red
    draw_image = SelectionModality.draw_rectangles(draw_image, pred_bboxes, color="red")

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
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Image saved to {file_name}")
