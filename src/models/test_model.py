from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from ..interfaces.base_custom_model import BaseCustomModel
from ..utils.consts import MODELS_PATH


@dataclass
class TestModel(BaseCustomModel):
    name = "test"

    def __post_init__(self) -> None:
        nn.Module.__init__(self)
        super().__post_init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def __hash__(self) -> int:
        return super().__hash__()


def train_model(model: BaseCustomModel, epochs: int) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config["learning_rate"])

    for epoch in range(1, epochs + 1):
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        if (epoch) % 5 == 0:
            model.save_checkpoint(epoch, optimizer, loss.item())


def test_base_custom_model() -> None:
    # Test creating a new model
    config = {"learning_rate": 0.01, "batch_size": 32}
    model = TestModel(config=config)
    print(f"Created new model with version number: {model.version_num}")

    # Train the model for a few epochs
    train_model(model, epochs=12)

    # Test loading the model from config
    loaded_model = TestModel.load_from_config(model.name, model.version_num)
    print(f"Loaded model configuration: {loaded_model.config}")

    # Test restarting from checkpoint
    optimizer = optim.Adam(
        loaded_model.parameters(), lr=loaded_model.config["learning_rate"]
    )
    start_epoch, best_loss, loaded_config = loaded_model.restart_from_checkpoint(
        optimizer
    )
    print(
        f"Restarted model {loaded_model.version} from epoch {start_epoch}, loss: {best_loss}"
    )
    print(f"Loaded configuration: {loaded_config}")

    # Continue training
    train_model(loaded_model, epochs=5)

    # Verify that a new version is created when we create another model
    another_model = TestModel(config=config)  # type: ignore
    print(f"Created another model with version number: {another_model.version_num}")

    # Clean up
    for path in MODELS_PATH.glob("TestModel_models"):
        for version_path in path.glob("version_*"):
            for file in version_path.glob("*"):
                file.unlink()
            version_path.rmdir()
        path.rmdir()


if __name__ == "__main__":
    test_base_custom_model()
