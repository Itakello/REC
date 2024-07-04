from dataclasses import dataclass, field

import clip
import torch
from itakello_logging import ItakelloLogging
from PIL import Image
from torchvision.transforms import Compose

from ..utils.consts import DEVICE

logger = ItakelloLogging().get_logger(__name__)

CLIP_MODEL = "RN50"


@dataclass
class CLIP:
    model: torch.nn.Module = field(init=False)
    preprocess: Compose = field(init=False)

    def __post_init__(self) -> None:
        self.model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)
        logger.debug("CLIP model loaded.")

    def encode_sentences(self, sentences: str | list[str]) -> torch.Tensor:
        # Tokenize the sentences
        text = clip.tokenize(sentences, truncate=True).to(DEVICE)

        # Encode the sentences using CLIP
        with torch.no_grad():
            features = self.model.encode_text(text)

        # L2 normalization
        features /= features.norm(dim=-1, keepdim=True)

        # Perform mean pooling
        features = features.mean(dim=0).float().to(DEVICE)
        logger.debug(f"Sentences encoded: {len(sentences)} | Shape: {features.shape}")
        return features

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        # Preprocess the images
        processed_images = [self.preprocess(image) for image in images]

        # Encode the images using CLIP
        images = torch.stack(processed_images).to(DEVICE)  # type: ignore
        with torch.no_grad():
            features = self.model.encode_image(images)

        # L2 normalization
        features /= features.norm(dim=-1, keepdim=True).float().to(DEVICE)

        logger.debug(f"Images encoded: {len(images)} | Shape: {features.shape}")
        return features
