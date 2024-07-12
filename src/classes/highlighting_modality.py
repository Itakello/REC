from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from ..datasets.yolo_baseline_dataset import YOLOBaselineDataset
from ..interfaces.base_class import BaseClass
from ..utils.consts import BLUR_INTENSITY, DATA_PATH, HIGHLIGHTING_METHODS, LINE_WIDTH
from ..utils.create_directory import create_directory


@dataclass
class HighlightingModality(BaseClass):
    @staticmethod
    def draw_rectangles(
        image: Image.Image,
        bboxes: list[list[float]] | torch.Tensor,
        color: str = "red",
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.tolist()

        if not isinstance(bboxes[0], list):
            bboxes = [bboxes]  # type: ignore

        for bbox in bboxes:
            draw.rectangle(bbox, outline=color, width=LINE_WIDTH)  # type: ignore

        return image

    @staticmethod
    def draw_circles(
        image: Image.Image,
        bboxes: list[list[float]] | torch.Tensor,
        color: str = "red",
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.tolist()

        if not isinstance(bboxes[0], list):
            bboxes = [bboxes]  # type: ignore

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radius = max(x2 - x1, y2 - y1) / 2  # type: ignore
            draw.ellipse(
                [
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ],
                outline=color,
                width=LINE_WIDTH,
            )

        return image

    @staticmethod
    def blur(
        image: Image.Image,
        bboxes: list[list[float]] | torch.Tensor,
    ) -> Image.Image:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.tolist()

        if not isinstance(bboxes[0], list):
            bboxes = [bboxes]  # type: ignore

        mask = Image.new("L", image.size, 255)
        draw = ImageDraw.Draw(mask)

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            draw.rectangle([x1, y1, x2, y2], fill=0)

        blurred = image.filter(ImageFilter.GaussianBlur(BLUR_INTENSITY))
        image.paste(blurred, mask=mask)

        return image

    @staticmethod
    def crop(
        image: Image.Image,
        bboxes: list[list[float]] | torch.Tensor,
    ) -> Image.Image:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.tolist()

        if not isinstance(bboxes[0], list):
            bboxes = [bboxes]  # type: ignore

        # We'll just crop to the first bounding box for this example
        bbox = bboxes[0]
        return image.crop(bbox)  # type: ignore

    @staticmethod
    def blackout(
        image: Image.Image,
        bboxes: list[list[float]] | torch.Tensor,
    ) -> Image.Image:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.tolist()

        if not isinstance(bboxes[0], list):
            bboxes = [bboxes]  # type: ignore

        # Create a black background
        blackout_image = Image.new("RGB", image.size, (0, 0, 0))

        # Create a mask for the areas we want to keep
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            draw.rectangle([x1, y1, x2, y2], fill=255)

        # Paste the original image onto the black background using the mask
        blackout_image.paste(image, (0, 0), mask)

        return blackout_image

    @staticmethod
    def apply_highlighting(
        image: Image.Image,
        bboxes: list[list[float]] | torch.Tensor,
        method: str,
    ) -> Image.Image:
        if method == "rectangle":
            return HighlightingModality.draw_rectangles(image, bboxes)
        elif method == "ellipse":
            return HighlightingModality.draw_circles(image, bboxes)
        elif method == "blur":
            return HighlightingModality.blur(image, bboxes)
        elif method == "crop":
            return HighlightingModality.crop(image, bboxes)
        elif method == "blackout":
            return HighlightingModality.blackout(image, bboxes)
        else:
            raise ValueError(f"Unknown highlighting method: {method}")


if __name__ == "__main__":

    # Initialize the YOLOBaselineDataset
    dataset = YOLOBaselineDataset(
        annotations_path=DATA_PATH / "annotations.csv",
        images_path=DATA_PATH / "images",
        embeddings_path=DATA_PATH / "embeddings",
        split="train",
        limit=1,
    )

    # Get the first sample from the dataset
    image = dataset[0]["image"]
    bbox = dataset[0]["bbox"]

    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle("Highlighting Modality Results")

    assert isinstance(axs, np.ndarray)
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    for i, method in enumerate(HIGHLIGHTING_METHODS, start=1):
        result = HighlightingModality.apply_highlighting(image.copy(), bbox, method)
        axs[i].imshow(result)
        axs[i].set_title(method.capitalize())
        axs[i].axis("off")

    plt.tight_layout()

    # Create 'samples' directory if it doesn't exist
    samples_dir = create_directory(Path("./samples"))

    # Save the figure
    plt.savefig(
        samples_dir / "highlighting_modalities_sample.jpg", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print("All highlighting methods tested and saved successfully!")
    print(
        f"Sample image saved at: {samples_dir / 'highlighting_modalities_sample.jpg'}"
    )
