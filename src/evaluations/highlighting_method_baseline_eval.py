from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from itakello_logging import ItakelloLogging
from tqdm import tqdm

import wandb

from ..classes.highlighting_modality import HighlightingModality
from ..classes.metric import Metrics
from ..datasets.highlighting_method_baseline_dataset import (
    HighlightingMethodBaselineDataset,
)
from ..interfaces.base_eval import BaseEval
from ..models.clip_model import ClipModel
from ..utils.consts import CLIP_MODEL, HIGHLIGHTING_METHODS, STATS_PATH, WANDB_PROJECT
from ..utils.create_directory import create_directory

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class HighlightingMethodEval(BaseEval):
    highlighting_methods: list[str] = field(
        default_factory=lambda: HIGHLIGHTING_METHODS
    )
    sentences_type: str = "combined_sentences"
    name: str = "highlighting-method"
    clip_model: ClipModel = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.clip_model = ClipModel(version=CLIP_MODEL)

    def evaluate(self) -> dict[str, Metrics]:
        results = {method: [] for method in self.highlighting_methods}
        dataloader = HighlightingMethodBaselineDataset.get_dataloaders()["val"]
        total_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating highlighting methods"):
            for (
                image,
                yolo_predictions,
                correct_candidate_idx,
                sentence_embedding,
            ) in zip(
                batch["images"],
                batch["yolo_predictions"],
                batch["correct_candidates_idx"],
                batch[f"{self.sentences_type}_embeddings"],
            ):
                total_samples += 1
                for method in self.highlighting_methods:
                    highlighted_images = [
                        HighlightingModality.apply_highlighting(
                            image.copy(), bbox, method
                        )
                        for bbox in yolo_predictions
                    ]
                    highlighted_embeddings = self.clip_model.encode_images(
                        highlighted_images
                    )

                    similarities = self.clip_model.get_similarity(
                        highlighted_embeddings, sentence_embedding
                    ).squeeze()

                    # Rank candidates based on similarities
                    _, sorted_indices = torch.sort(similarities, descending=True)

                    # Find the rank of the correct candidate
                    correct_candidate_rank = torch.where(
                        sorted_indices == correct_candidate_idx
                    )[0].item()
                    results[method].append(correct_candidate_rank)

        # Calculate cumulative distributions and create Metrics objects
        metrics = {}
        for method, ranks in results.items():
            hist, _ = np.histogram(ranks, bins=range(11))  # 10 ranks (1-10)
            cumulative_dist = np.cumsum(hist) / total_samples

            metrics[method] = Metrics()
            for rank, value in enumerate(cumulative_dist, start=1):
                metrics[method].add(f"{rank}", value)

        self.plot_cumulative_distributions(metrics)
        self.log_metrics(metrics)

        return metrics

    def plot_cumulative_distributions(self, metrics: dict[str, Metrics]) -> None:
        plt.figure(figsize=(12, 8))
        for method, method_metrics in metrics.items():
            dist = [metric.value for metric in method_metrics]
            plt.plot(range(1, 11), dist, marker="o", label=method)

        plt.xlabel("Rank")
        plt.ylabel("Cumulative Proportion")
        plt.title("Cumulative Rank Distribution")
        plt.legend()
        plt.grid(True)

        plt.ylim(0, 1)

        # Set y-axis to display proportions
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{100*y:.0f}%")  # type: ignore
        )

        # Save the plot
        plot_path = (
            create_directory(STATS_PATH)
            / f"{self.name}_cumulative_rank_distribution.png"
        )
        plt.savefig(plot_path)
        plt.close()

        logger.confirmation(f"Cumulative rank distribution plot saved to {plot_path}")

    def log_metrics(self, metrics: Metrics | dict[str, Metrics]) -> None:
        assert isinstance(metrics, dict)
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{self.name}_cumulative_rank_distribution",
        )

        for method, method_metrics in metrics.items():
            for metric in method_metrics:
                wandb.log({f"{method}_{metric.name}": metric.value})

        # Log the plot
        plot_path = STATS_PATH / f"{self.name}_cumulative_rank_distribution.png"
        wandb.log({"cumulative_rank_distribution": wandb.Image(str(plot_path))})

        run.finish()


if __name__ == "__main__":
    evaluator = HighlightingMethodEval()
    metrics = evaluator.evaluate()
    print("Evaluation completed. Results saved and plotted.")
    for method, method_metrics in metrics.items():
        print(f"\n{method}:")
        for metric in method_metrics:
            print(f"  {metric.name}: {metric.value:.4f}")
