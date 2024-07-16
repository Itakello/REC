import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
from itakello_logging import ItakelloLogging
from torch.utils.data import DataLoader

import wandb

from ..classes.metric import Metrics
from ..datasets.refcocog_base_dataset import RefCOCOgBaseDataset
from ..interfaces.base_class import BaseClass
from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE, WANDB_PROJECT

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class TrainerManager(BaseClass):
    model_name: str
    config_path: Path
    model: BaseCustomModel
    dataset_cls: Type[RefCOCOgBaseDataset]
    loss_fn: nn.Module | None = None
    config: dict = field(init=False)
    dataloaders: dict[str, DataLoader] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.config = self._load_config()
        self.model.to(DEVICE)
        self.dataloaders = self._create_dataloaders()

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as file:
            config = json.load(file)
        return config[self.model_name]

    def _create_dataloaders(self) -> dict[str, DataLoader]:
        batch_size = self.config["batch_sizes"][0]
        return self.dataset_cls.get_dataloaders(batch_size=batch_size)

    def _get_optimizer(
        self, optimizer_config: dict, lr: float
    ) -> torch.optim.Optimizer:
        optimizer_class = getattr(torch.optim, optimizer_config["type"])
        return optimizer_class(
            self.model.parameters(),
            lr=lr,
            **{k: v for k, v in optimizer_config.items() if k != "type"},
        )

    def _get_loss_function(self) -> nn.Module:
        if self.loss_fn is not None:
            return self.loss_fn
        loss_name = self.config["loss"][0]
        if hasattr(nn, loss_name):
            return getattr(nn, loss_name)()
        else:
            custom_loss_module = importlib.import_module("custom_losses")
            if hasattr(custom_loss_module, loss_name):
                return getattr(custom_loss_module, loss_name)()
            else:
                raise ValueError(
                    f"Loss function {loss_name} not found in torch.nn or custom_losses module."
                )

    def train(
        self, epochs: int, use_combinations: bool = False, from_checkpoint: bool = False
    ) -> None:
        if use_combinations:
            for lr in self.config["learning_rates"]:
                for optimizer_config in self.config["optimizers"]:
                    for batch_size in self.config["batch_sizes"]:
                        run_config = {
                            "learning_rate": lr,
                            "optimizer": optimizer_config["type"],
                            "batch_size": batch_size,
                            "loss": self.config["loss"][0],
                            "weight_decay": self.config["weight_decay"][0],
                        }
                        with wandb.init(
                            project=WANDB_PROJECT,
                            name=f"{self.model.name}_training",
                            config=run_config,
                        ):
                            self._train_single_configuration(
                                epochs,
                                lr,
                                optimizer_config,
                                batch_size,
                                from_checkpoint,
                            )
        else:
            run_config = {
                "learning_rate": self.config["learning_rates"][0],
                "optimizer": self.config["optimizers"][0]["type"],
                "batch_size": self.config["batch_sizes"][0],
                "loss": self.config["loss"][0],
                "weight_decay": self.config["weight_decay"][0],
            }
            with wandb.init(
                project=WANDB_PROJECT,
                name=f"{self.model.name}_training",
                config=run_config,
            ):
                self._train_single_configuration(
                    epochs,
                    self.config["learning_rates"][0],
                    self.config["optimizers"][0],
                    self.config["batch_sizes"][0],
                    from_checkpoint,
                )

    def _train_single_configuration(
        self,
        epochs: int,
        lr: float,
        optimizer_config: dict,
        batch_size: int,
        from_checkpoint: bool,
    ) -> None:
        self.model.train()
        optimizer = self._get_optimizer(optimizer_config, lr)
        criterion = self._get_loss_function()
        self.dataloaders = self.dataset_cls.get_dataloaders(batch_size=batch_size)

        start_epoch = 0
        if from_checkpoint:
            start_epoch, _, _ = self.model.restart_from_checkpoint(optimizer)
            logger.info(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            train_metrics = self._train_epoch(optimizer, criterion)
            self._log_metrics(epoch, train_metrics, "train")

            val_metrics = self.evaluate()
            self._log_metrics(epoch, val_metrics, "val")

            if (epoch + 1) % 5 == 0:
                self.model.save_checkpoint(epoch, optimizer, train_metrics["loss"])

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> Metrics:
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.dataloaders["train"]:
            inputs, labels = self.model.prepare_input(batch)

            optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += self.model.calculate_accuracy(outputs, labels)
            total += labels.size(0)

        metrics = Metrics()
        metrics.add("avg_loss", total_loss / len(self.dataloaders["train"]))
        metrics.add("accuracy", 100 * correct / total)
        return metrics

    def evaluate(self) -> Metrics:
        self.model.eval()
        metrics = Metrics()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.dataloaders["val"]:
                inputs, labels = self.model.prepare_input(batch)
                outputs = self.model(**inputs)
                correct += self.model.calculate_accuracy(outputs, labels)
                total += labels.size(0)

        metrics.add("accuracy", 100 * correct / total)
        return metrics

    def _log_metrics(self, epoch: int, metrics: Metrics, split: str) -> None:
        wandb.log(
            {f"{split}/{metric.name}": metric.value for metric in metrics}
            | {"epoch": epoch}
        )

        logger.info(f"Epoch [{epoch + 1}] - {split.capitalize()}:\n{metrics}")


if __name__ == "__main__":
    from ..datasets.classification_v0_dataset import ClassificationV0Dataset
    from ..models.classification_v0_model import ClassificationV0Model
    from ..utils.consts import CONFIG_PATH

    model = ClassificationV0Model()
    dataset_cls = ClassificationV0Dataset

    trainer = TrainerManager(
        model_name=model.name,
        config_path=CONFIG_PATH / "trainer_config.json",
        model=model,
        dataset_cls=dataset_cls,
    )

    trainer.train(epochs=5, use_combinations=True)
