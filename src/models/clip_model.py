from dataclasses import dataclass, field

import clip
import torch
from itakello_logging import ItakelloLogging
from PIL import Image
from torchvision.transforms import Compose

from ..interfaces.base_model import BaseModel
from ..utils.consts import CLIP_MODEL, DEVICE

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class ClipModel(BaseModel):
    name: str = "clip"
    model: torch.nn.Module = field(init=False)
    preprocess: Compose = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model, self.preprocess = clip.load(
            str(self.version), device=DEVICE, download_root=str(self.model_path)
        )

    def encode_sentences(
        self, sentences: str | list[str], average: bool = False
    ) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        # Tokenize the sentences
        text = clip.tokenize(sentences, truncate=True).to(DEVICE)

        # Encode the sentences using CLIP
        with torch.no_grad():
            text_features = self.model.encode_text(text).to(DEVICE)

        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)

        # Average the features if there are multiple sentences
        if average:
            text_features = text_features.mean(dim=0, keepdim=True)

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
            image_features = self.model.encode_image(image_input).to(DEVICE)

        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        return image_features

    def get_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text features.

        Args:
            image_features: Tensor of shape (N, D) or (B, N, D)
                where N is the number of images, D is the feature dimension,
                and B is an optional batch dimension.
            text_features: Tensor of shape (M, D) or (B, M, D)
                where M is the number of texts, D is the feature dimension,
                and B is an optional batch dimension.

        Returns:
            Tensor of shape (N, M) or (B, N, M) containing cosine similarities.
        """
        # Ensure inputs are at least 2D
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)

        # Normalize featyures
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if image_features.dim() == 3 and text_features.dim() == 3:
            # Batch processing
            # Compute cosine similarity
            similarity = torch.bmm(image_features, text_features.transpose(1, 2))
        else:
            # Non-batch processing
            # Compute cosine similarity
            similarity = torch.mm(image_features, text_features.T)

        return similarity


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from ..datasets.refcocog_base_dataset import RefCOCOgBaseDataset

    # Initialize CLIP
    clip_model = ClipModel(version=CLIP_MODEL)

    # Initialize the dataset
    dataset = RefCOCOgBaseDataset()

    # Create a DataLoader
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn
    )

    # Test different scenarios
    for batch in dataloader:
        # 1. Single image, single text
        single_image = batch["images"][0]
        single_text = batch["comprehensive_sentences"][0]

        single_image_feature = clip_model.encode_images(single_image)
        single_text_feature = clip_model.encode_sentences(single_text)

        single_similarity = clip_model.get_similarity(
            single_image_feature, single_text_feature
        )
        print(
            "Single image, single text similarity shape:", single_similarity.shape
        )  # Expected: [1, 1]

        # 2. Multiple images, single text
        multi_image_features = clip_model.encode_images(batch["images"])
        multi_single_similarity = clip_model.get_similarity(
            multi_image_features, single_text_feature
        )
        print(
            "Multiple images, single text similarities shape:",
            multi_single_similarity.shape,
        )  # Expected: [2, 1]

        # 3a. Single image, multiple texts (same sample)
        single_image_feature = clip_model.encode_images(batch["images"][0])
        same_sample_texts = batch["sentences"][0]  # Multiple texts
        same_sample_text_features = clip_model.encode_sentences(
            same_sample_texts, average=True
        )
        single_multi_same_similarity = clip_model.get_similarity(
            single_image_feature, same_sample_text_features
        )
        print(
            "Single image, multiple texts (same sample) similarities shape:",
            single_multi_same_similarity.shape,
        )  # Expected: [1, 1]

        # 3b. Single image, multiple texts (different samples)
        different_sample_text_features = clip_model.encode_sentences(
            [batch["comprehensive_sentences"][0], batch["comprehensive_sentences"][1]]
        )
        single_multi_diff_similarity = clip_model.get_similarity(
            single_image_feature, different_sample_text_features
        )
        print(
            "Single image, multiple texts (different samples) similarities shape:",
            single_multi_diff_similarity.shape,
        )  # Expected: [1, 2]

        # 4. Multiple images, multiple texts (batch processing)
        multi_text_features = clip_model.encode_sentences(
            batch["comprehensive_sentences"]
        )
        batch_similarities = clip_model.get_similarity(
            multi_image_features, multi_text_features
        )
        print("Batch similarities shape:", batch_similarities.shape)  # Expected: [2, 2]

        # 5. Test with embeddings from the dataset
        image_embeddings = torch.stack(
            [torch.from_numpy(emb["image_features"]) for emb in batch["embeddings"]]
        )
        text_embeddings = torch.stack(
            [
                torch.from_numpy(emb["comprehensive_sentence_features"])
                for emb in batch["embeddings"]
            ]
        )
        image_embeddings = image_embeddings.squeeze(1)
        text_embeddings = text_embeddings.squeeze(1)

        dataset_similarities = clip_model.get_similarity(
            image_embeddings, text_embeddings
        )
        print(
            "Dataset embeddings similarities shape:", dataset_similarities.shape
        )  # Expected: [2, 2]
        print("Image embeddings shape:", image_embeddings.shape)  # [2, 1024]
        print("Text embeddings shape:", text_embeddings.shape)  # [2, 1024]

        # Only process one batch for this test
        break

    print("All tests completed successfully!")
