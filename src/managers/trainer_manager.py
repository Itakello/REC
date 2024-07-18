import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
from itakello_logging import ItakelloLogging
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ..classes.metric import Metrics
from ..datasets.refcocog_base_dataset import RefCOCOgBaseDataset
from ..interfaces.base_class import BaseClass
from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import DEVICE, WANDB_PROJECT

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class TrainerManager(BaseClass):
    model_class: Type[BaseCustomModel]
    config_path: Path
    dataset_cls: Type[RefCOCOgBaseDataset]
    dataset_limit: int = -1
    is_regression: bool = False
    config: dict = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.config = self._load_config()

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as file:
            config = json.load(file)
        return config[self.model_class.name]

    def _get_dataloaders(self, batch_size: int) -> dict[str, DataLoader]:
        return self.dataset_cls.get_dataloaders(
            batch_size=batch_size, limit=self.dataset_limit
        )

    def _get_optimizer(
        self, model: BaseCustomModel, optimizer_config: dict, lr: float
    ) -> torch.optim.Optimizer:
        optimizer_class = getattr(torch.optim, optimizer_config["type"])
        return optimizer_class(
            model.parameters(),
            lr=lr,
            **{k: v for k, v in optimizer_config.items() if k != "type"},
        )

    def _get_loss_function(self, loss_name: str) -> nn.Module:
        if hasattr(nn, loss_name):
            return getattr(nn, loss_name)()
        else:
            custom_loss_module = importlib.import_module("src.classes.custom_losses")
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
            self._train_multiple_configurations(epochs, from_checkpoint)
        else:
            run_config = {
                "learning_rate": self.config["learning_rates"][0],
                "optimizer": self.config["optimizers"][0]["type"],
                "batch_size": self.config["batch_sizes"][0],
                "loss": self.config["losses"][0],
            }
            model = self.model_class(config=run_config)
            model.to(DEVICE)
            with wandb.init(
                project=WANDB_PROJECT,
                name=f"{model.name}_v{model.version_num}",
                config=run_config,
            ):
                self._train_single_configuration(
                    model,
                    epochs,
                    self.config["learning_rates"][0],
                    self.config["optimizers"][0],
                    self.config["batch_sizes"][0],
                    self.config["loss"][0],
                    from_checkpoint,
                )

    def _train_multiple_configurations(
        self, epochs: int, from_checkpoint: bool
    ) -> None:
        for lr in self.config["learning_rates"]:
            for optimizer_config in self.config["optimizers"]:
                for batch_size in self.config["batch_sizes"]:
                    for loss_name in self.config["losses"]:
                        run_config = {
                            "learning_rate": lr,
                            "optimizer": optimizer_config["type"],
                            "batch_size": batch_size,
                            "loss": loss_name,
                        }
                        model = self.model_class(config=run_config)
                        model.to(DEVICE)
                        with wandb.init(
                            project=WANDB_PROJECT,
                            name=f"{model.name}_v{model.version_num}",
                            config=run_config,
                        ):
                            self._train_single_configuration(
                                model,
                                epochs,
                                lr,
                                optimizer_config,
                                batch_size,
                                loss_name,
                                from_checkpoint,
                            )

    def _train_single_configuration(
        self,
        model: BaseCustomModel,
        epochs: int,
        lr: float,
        optimizer_config: dict,
        batch_size: int,
        loss_name: str,
        from_checkpoint: bool,
    ) -> None:
        model.train()
        optimizer = self._get_optimizer(model, optimizer_config, lr)
        criterion = self._get_loss_function(loss_name)
        dataloaders = self._get_dataloaders(batch_size)

        global_steps = 0
        start_epoch = 0
        if from_checkpoint:
            start_epoch, _, _ = model.restart_from_checkpoint(optimizer)
            logger.info(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            train_metrics, global_steps = self._train_epoch(
                model, optimizer, criterion, dataloaders["train"], global_steps
            )
            self._log_metrics(global_steps, train_metrics, "train")

            val_metrics = self.evaluate(model, "val", dataloaders, criterion)
            self._log_metrics(global_steps, val_metrics, "val")

            if (epoch + 1) % 10 == 0:
                test_metrics = self.evaluate(model, "test", dataloaders, criterion)
                self._log_metrics(global_steps, test_metrics, "test")
                model.save_checkpoint(epoch, optimizer, train_metrics["loss"])

    def _train_epoch(
        self,
        model: BaseCustomModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        global_steps: int,
    ) -> tuple[Metrics, int]:
        total_loss = 0.0
        total_spatial_loss = 0.0
        total_semantic_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_dataloader, desc="Train epoch", unit="batch"):
            inputs, labels = model.prepare_input(batch)

            optimizer.zero_grad()
            if self.is_regression:
                if model.name == "regression_v1":
                    outputs = model(*inputs[1:-2])
                elif model.name == "regression_v0":
                    outputs = model(*inputs[:-2])
                else:
                    raise ValueError("Model name not recognized.")
                loss, spatial_loss, semantic_loss = criterion(
                    outputs, labels, inputs[4], inputs[1], inputs[5]
                )
                total_spatial_loss += spatial_loss.item()
                total_semantic_loss += semantic_loss.item()
            else:
                outputs = model(*inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()

            batch_metrics = Metrics()
            batch_metrics.add("batch_loss", batch_loss)
            if self.is_regression:
                batch_metrics.add("batch_spatial_loss", spatial_loss.item())
                batch_metrics.add("batch_semantic_loss", semantic_loss.item())
            self._log_metrics(global_steps, batch_metrics, "train", False)

            total_loss += batch_loss
            correct += model.calculate_accuracy(outputs, labels)
            total += labels.size(0)
            global_steps += 1

        metrics = Metrics()
        metrics.add("loss", total_loss / len(train_dataloader))
        if self.is_regression:
            metrics.add("spatial_loss", total_spatial_loss / len(train_dataloader))
            metrics.add("semantic_loss", total_semantic_loss / len(train_dataloader))
        metrics.add("accuracy", 100 * correct / total)
        return metrics, global_steps

    def evaluate(
        self,
        model: BaseCustomModel,
        split: str,
        dataloaders: dict[str, DataLoader],
        criterion: nn.Module,
    ) -> Metrics:
        model.eval()
        metrics = Metrics()
        total_loss = 0.0
        total_spatial_loss = 0.0
        total_semantic_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(
                dataloaders[split], desc=f"Evaluating {split}", unit="batch"
            ):
                inputs, labels = model.prepare_input(batch)
                if self.is_regression:
                    outputs = model(*inputs[:-2])
                    loss, spatial_loss, semantic_loss = criterion(
                        outputs, labels, inputs[4], inputs[1], inputs[5]
                    )
                    total_spatial_loss += spatial_loss.item()
                    total_semantic_loss += semantic_loss.item()
                else:
                    outputs = model(*inputs)
                    loss = criterion(outputs, labels)
                total_loss += loss.item()
                correct += model.calculate_accuracy(outputs, labels)
                total += labels.size(0)

        metrics.add("loss", total_loss / len(dataloaders[split]))
        if self.is_regression:
            metrics.add("spatial_loss", total_spatial_loss / len(dataloaders[split]))
            metrics.add("semantic_loss", total_semantic_loss / len(dataloaders[split]))
        metrics.add("accuracy", 100 * correct / total)
        return metrics

    def _log_metrics(
        self, epoch: int, metrics: Metrics, split: str, show_log: bool = False
    ) -> None:
        wandb.log(
            data={f"{split}/{metric.name}": metric.value for metric in metrics},
            step=epoch,
        )
        if show_log:
            logger.info(f"Epoch [{epoch}] - {split.capitalize()}:\n{metrics}")


if __name__ == "__main__":
    from ..datasets.classification_dataset import ClassificationDataset
    from ..datasets.regression_dataset import RegressionDataset
    from ..models.classification_v0_model import ClassificationV0Model
    from ..models.regression_v0_model import RegressionV0Model
    from ..utils.consts import CONFIG_PATH

    trainer = TrainerManager(
        model_class=RegressionV0Model,
        config_path=CONFIG_PATH / "trainer_config.json",
        dataset_cls=RegressionDataset,
        dataset_limit=20000,
        is_regression=True,
    )

    trainer.train(epochs=10, use_combinations=True)
