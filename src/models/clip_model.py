from dataclasses import dataclass, field
from pathlib import Path

import clip
import torch
from itakello_logging import ItakelloLogging
from PIL import Image
from torchvision.transforms import Compose

from ..interfaces.base_model import BaseModel
from ..utils.consts import CLIP_MODEL, DEVICE, MODELS_PATH

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class ClipModel(BaseModel):
    name: str = "clip"
    model: torch.nn.Module = field(init=False)
    preprocess: Compose = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model, self.preprocess = clip.load(
            self.version, device=DEVICE, download_root=str(self.model_path)
        )

    def encode_sentences(self, sentences: str | list[str]) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        # Tokenize the sentences
        text = clip.tokenize(sentences, truncate=True).to(DEVICE)

        # Encode the sentences using CLIP
        with torch.no_grad():
            text_features = self.model.encode_text(text)

        text_features = text_features.float().to(DEVICE)
        logger.debug(
            f"Sentences encoded: {len(sentences)} | Shape: {text_features.shape}"
        )
        return text_features

    def encode_images(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        # Preprocess and stack the images
        image_input = torch.stack([self.preprocess(image) for image in images]).to(  # type: ignore
            DEVICE
        )

        # Encode the images using CLIP
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        image_features = image_features.float().to(DEVICE)
        logger.debug(f"Images encoded: {len(images)} | Shape: {image_features.shape}")
        return image_features

    def get_similarity(
        self, images: Image.Image | list[Image.Image], texts: str | list[str]
    ) -> torch.Tensor:
        image_features = self.encode_images(images)
        text_features = self.encode_sentences(texts)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = image_features @ text_features.T

        return similarity


if __name__ == "__main__":
    # Initialize CLIP
    clip_model = ClipModel(version=CLIP_MODEL, models_path=MODELS_PATH)

    # Define image paths and sentences
    image_dir = Path("data/images")
    image_paths = [
        image_dir / "COCO_train2014_000000352312.jpg",
        image_dir / "COCO_train2014_000000426640.jpg",
    ]
    sentences = [
        "A polar bear is pondering going down the ramp and into the water.",
        "big bear",
    ]

    # Test single image encoding
    single_image = Image.open(image_paths[0])
    single_image_features = clip_model.encode_images(single_image)
    print(f"Single image features shape: {single_image_features.shape}")

    # Test multiple image encoding
    images = [Image.open(path) for path in image_paths]
    multi_image_features = clip_model.encode_images(images)  # type: ignore
    print(f"Multiple image features shape: {multi_image_features.shape}")

    # Test single sentence encoding
    single_sentence_features = clip_model.encode_sentences(sentences[0])
    print(f"Single sentence features shape: {single_sentence_features.shape}")

    # Test multiple sentence encoding
    multi_sentence_features = clip_model.encode_sentences(sentences)
    print(f"Multiple sentence features shape: {multi_sentence_features.shape}")

    # Test similarity
    # Compare sentences with first image
    similarity_first = clip_model.get_similarity(images[0], sentences)
    print(f"Similarity with first image:\n{similarity_first}")

    # Compare sentences with second image
    similarity_second = clip_model.get_similarity(images[1], sentences)
    print(f"Similarity with second image:\n{similarity_second}")

    # Compare sentences with both images
    similarity_both = clip_model.get_similarity(images, sentences)  # type: ignore
    print(f"Similarity with both images:\n{similarity_both}")

    # Check if sentences are more similar to the first image
    is_more_similar = torch.all(similarity_both[0] > similarity_both[1])
    print(f"Sentences are more similar to the first image: {is_more_similar}")

    # Print the difference in similarities
    similarity_diff = similarity_both[0] - similarity_both[1]
    print(f"Difference in similarities (first - second):\n{similarity_diff}")
