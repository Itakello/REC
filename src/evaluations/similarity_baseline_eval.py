from dataclasses import dataclass, field

from itakello_logging import ItakelloLogging
from torch.utils.data import DataLoader

import wandb

from ..classes.highlighting_modality import HighlightingModality
from ..classes.metric import Metrics
from ..datasets.similarity_baseline_dataset import SimilarityBaselineDataset
from ..interfaces.base_eval import BaseEval
from ..models.clip_model import ClipModel
from ..utils.consts import CLIP_MODEL, DATA_PATH, MODELS_PATH, WANDB_PROJECT

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class SimilarityBaselineEval(BaseEval):
    highlighting_methods: list[str] = field(default_factory=list)
    sentences_types: list[str] = field(default_factory=list)
    name: str = "similarity-baseline"
    dataset: SimilarityBaselineDataset = field(init=False)
    clip_model: ClipModel = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dataset = SimilarityBaselineDataset(
            annotations_path=DATA_PATH / "annotations.csv",
            images_path=DATA_PATH / "images",
            embeddings_path=DATA_PATH / "embeddings",
        )
        self.clip_model = ClipModel(version=CLIP_MODEL, models_path=MODELS_PATH)

    def get_dataloaders(self) -> list[tuple[str, DataLoader]]:
        return [
            (
                "test",
                DataLoader(
                    self.dataset, collate_fn=self.dataset.collate_fn, shuffle=False
                ),
            )
        ]

    def evaluate(self) -> dict[str, Metrics]:
        results = {
            method: {sentence_type: [] for sentence_type in self.sentences_types}
            for method in self.highlighting_methods
        }

        for batch in self.get_dataloaders()[0][1]:
            images = batch["images"]
            bboxes = batch["bboxes"]
            original_sentences_embeddings = batch["original_sentences_embeddings"]
            comprehensive_sentence_embeddings = batch[
                "comprehensive_sentence_embeddings"
            ]
            combined_sentences_embeddings = batch["combined_sentences_embeddings"]

            for (
                image,
                bbox,
                original_sentence_embedding,
                comprehensive_sentence_embedding,
                combined_sentence_embedding,
            ) in zip(
                images,
                bboxes,
                original_sentences_embeddings,
                comprehensive_sentence_embeddings,
                combined_sentences_embeddings,
            ):
                for method in self.highlighting_methods:
                    highlighted_image = HighlightingModality.apply_highlighting(
                        image, bbox, method
                    )
                    highlighted_embedding = self.clip_model.encode_images(
                        highlighted_image
                    )

                    original_similarity = self.clip_model.get_similarity(
                        highlighted_embedding, original_sentence_embedding
                    )
                    comprehensive_similarity = self.clip_model.get_similarity(
                        highlighted_embedding, comprehensive_sentence_embedding
                    )
                    combined_similarity = self.clip_model.get_similarity(
                        highlighted_embedding, combined_sentence_embedding
                    )

                    results[method]["original_sentences"].append(
                        original_similarity.item()
                    )
                    results[method]["comprehensive_sentence"].append(
                        comprehensive_similarity.item()
                    )
                    results[method]["combined_sentences"].append(
                        combined_similarity.item()
                    )

        # Normalize results
        metrics = {}
        for method in self.highlighting_methods:
            metrics[method] = Metrics()
            for sentence_type in self.sentences_types:
                avg_similarity = sum(results[method][sentence_type]) / len(
                    results[method][sentence_type]
                )
                metrics[method].add(f"{sentence_type}_similarity", avg_similarity)

        self.log_metrics(metrics)
        logger.confirmation("Similarity baseline evaluation completed")
        return metrics

    def log_metrics(self, metrics: Metrics | dict[str, Metrics]) -> None:
        assert isinstance(metrics, dict)
        for method, method_metrics in metrics.items():
            run = wandb.init(
                project=WANDB_PROJECT,
                name=f"{method}",
            )
            wandb.log(
                {
                    f"{self.name}/{metric.name}": metric.value
                    for metric in method_metrics
                },
            )
            run.finish()


if __name__ == "__main__":
    from ..utils.consts import HIGHLIGHTING_METHODS, SENTENCES_TYPES

    evaluator = SimilarityBaselineEval(
        highlighting_methods=HIGHLIGHTING_METHODS, sentences_types=SENTENCES_TYPES
    )
    metrics = evaluator.evaluate()
    print("Evaluation metrics:")
    for method, method_metrics in metrics.items():
        print(f"{method}:")
        print(method_metrics)
