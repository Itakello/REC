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
        batch_size = self.config["default"]["batch_size"]
        return self.dataset_cls.get_dataloaders(batch_size=batch_size)

    def _get_optimizer(self, optimizer_config: dict) -> torch.optim.Optimizer:
        optimizer_class = getattr(torch.optim, optimizer_config["type"])
        return optimizer_class(
            self.model.parameters(),
            lr=self.config["default"]["learning_rate"],
            **{k: v for k, v in optimizer_config.items() if k != "type"},
        )

    def _get_loss_function(self) -> nn.Module:
        if self.loss_fn is not None:
            return self.loss_fn
        loss_name = self.config["default"]["loss"]
        if hasattr(nn, loss_name):
            return getattr(nn, loss_name)()
        else:
            # import custom loss from custom_losses module ... to complete!
            custom_loss_module = importlib.import_module("custom_losses")
            if hasattr(custom_loss_module, loss_name):
                return getattr(custom_loss_module, loss_name)()
            else:
                raise ValueError(
                    f"Loss function {loss_name} not found in torch.nn or custom_losses module."
                )

    def train(self, epochs: int, from_checkpoint: bool = False) -> None:
        optimizer = self._get_optimizer(self.config["default"]["optimizer"])
        criterion = self._get_loss_function()

        start_epoch = 0
        if from_checkpoint:
            start_epoch, _, _ = self.model.restart_from_checkpoint(optimizer)
            logger.info(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_metrics = self._train_epoch(optimizer, criterion)
            val_metrics = self.evaluate()

            self._log_metrics(epoch, train_metrics, val_metrics)
            if epoch + 1 % 5 == 0:
                self.model.save_checkpoint(epoch, optimizer, train_metrics["loss"])

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> Metrics:
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.dataloaders["train"]:
            # todo MODIFY WITH DATASET SPECIFIC INPUTS
            # inputs = self._prepare_inputs(batch)
            # labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = criterion(outputs, labels)
            # TODO save loss in metrics

            # todo CHECK THEM
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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
                # todo MODIFY WITH DATASET SPECIFIC INPUTS
                # inputs = self._prepare_inputs(batch)
                # labels = batch["labels"].to(DEVICE)

                outputs = self.model(**inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        metrics.add("accuracy", 100 * correct / total)
        return metrics

    def _log_metrics(
        self, epoch: int, train_metrics: Metrics, val_metrics: Metrics
    ) -> None:
        wandb.log(
            {f"train/{metric.name}": metric.value for metric in train_metrics}
            | {f"val/{metric.name}": metric.value for metric in val_metrics}
            | {"epoch": epoch}
        )

        logger.info(
            f"Epoch [{epoch + 1}]\n"
            + f"Train:\n{train_metrics}\n"
            + f"nVal:\n {val_metrics}"
        )


if __name__ == "__main__":
    from ..models.classification_v0_model import ClassificationV0Model
    from ..utils.consts import CONFIG_PATH

    model = ClassificationV0Model()
    dataset_cls = RefCOCOgBaseDataset

    trainer = TrainerManager(
        model_name=model.name,
        config_path=CONFIG_PATH / "trainer_config.json",
        model=model,
        dataset_cls=dataset_cls,
    )

    wandb.init(project=WANDB_PROJECT, name=f"{model.name}_training")
    trainer.train(epochs=5)
    wandb.finish()
