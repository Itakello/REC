from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw

from ..interfaces.base_class import BaseClass
from ..utils.consts import LINE_WIDTH


@dataclass
class SelectionModality(BaseClass):
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
