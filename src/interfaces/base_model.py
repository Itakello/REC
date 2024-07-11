from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.create_directory import create_directory
from .base_class import BaseClass


@dataclass
class BaseModel(BaseClass, ABC):
    version: str
    models_path: Path
    model_path: Path = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_path = create_directory(self.models_path / f"{self.name}_models")