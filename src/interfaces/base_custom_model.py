import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..utils.consts import DEVICE, MODELS_PATH
from ..utils.create_directory import create_directory
from .base_model import BaseModel


@dataclass(eq=False)
class BaseCustomModel(BaseModel, nn.Module, ABC):
    configuration: dict = field(default_factory=dict)
    version_path: Path = field(init=False)
    config_file_name: str = "config.json"
    from_checkpoint: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        model_path = create_directory(MODELS_PATH / f"{self.name}_models")
        if not self.from_checkpoint:
            self.version_num = self._get_next_version_num(model_path)
            self.version_path = create_directory(model_path / f"v{self.version_num}")
            self._save_configuration()

    def _get_next_version_num(self, base_path: Path) -> int:
        existing_configs = [
            int(p.name[1:])
            for p in base_path.glob("v*")
            if p.is_dir() and p.name.startswith("v")
        ]
        return max(existing_configs, default=0) + 1

    def _save_configuration(self) -> None:
        """
        Save the configuration settings to a JSON file in the model directory.
        """
        config_file = self.version_path / self.config_file_name
        with open(config_file, "w") as f:
            json.dump(self.configuration, f, indent=4)

    def _load_configuration(self) -> dict:
        """
        Load the configuration settings from the JSON file in the model directory.
        """
        config_file = self.version_path / self.config_file_name
        if not config_file.exists():
            raise FileNotFoundError(f"No configuration file found at {config_file}")

        with open(config_file, "r") as f:
            return json.load(f)

    def save_checkpoint(
        self, epoch: int, optimizer: torch.optim.Optimizer, loss: float
    ) -> None:
        """
        Save the model's checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(
            checkpoint,
            self.version_path / f"checkpoint_epoch_{epoch}.pt",
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> dict:
        """
        Load a checkpoint and return the checkpoint data.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _get_latest_checkpoint(self) -> Path | None:
        """
        Get the path of the latest checkpoint in the model's directory.
        """
        checkpoints = list(self.version_path.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))

    def restart_from_checkpoint(
        self, optimizer: torch.optim.Optimizer
    ) -> tuple[int, float, dict]:
        """
        Restart the model from the latest checkpoint if available.
        Returns the starting epoch, loss, and configuration.
        """
        latest_checkpoint = self._get_latest_checkpoint()
        if latest_checkpoint is None:
            return 0, float("inf"), self.configuration

        checkpoint_data = self._load_checkpoint(latest_checkpoint)
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.configuration = self._load_configuration()
        return checkpoint_data["epoch"], checkpoint_data["loss"], self.configuration

    @classmethod
    def load_from_config(cls, name: str, config_num: int) -> "BaseCustomModel":
        """
        Load a model from a specific configuration number.
        """
        version = f"v{config_num}"
        model_path = MODELS_PATH / f"{name}_models" / version
        if not model_path.exists():
            raise FileNotFoundError(f"No configuration found at {model_path}")

        with open(model_path / cls.config_file_name, "r") as f:
            configuration = json.load(f)

        model = cls(configuration=configuration, from_checkpoint=True)  # type: ignore
        model.version_num = config_num
        model.version_path = model_path
        model.version = version

        latest_checkpoint = model._get_latest_checkpoint()
        if latest_checkpoint:
            model._load_checkpoint(latest_checkpoint)

        return model

    def __hash__(self) -> int:
        return hash((type(self), self.version_num))
