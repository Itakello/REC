import json
import itertools
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from typing import Any, Dict, Type, Optional
from itakello_logging import ItakelloLogging


from ..datasets.refcocog_base_dataset import RefCOCOgBaseDataset 
from ..interfaces.base_custom_model import BaseCustomModel


logger = ItakelloLogging().get_logger(__name__)


"""
TODO: 
- turn print and stuff to Itakello-based logs. 
- adapt train/eval loop to all the architectures and respective inputs. 


example of json: 

{
    "Classification_v0": {
      "default": {
        "learning_rate": 0.002,
        "optimizer": {
          "type": "Adam",
          "betas": [0.9, 0.999]
        },
        "weight_decay": 0.0001,
        "batch_size": 64,
        "loss": "CrossEntropyLoss"
      },
      "combinations": {
        "learning_rates": [0.0001, 0.0002],
        "optimizers": [
          {"type": "Adam", "betas": [0.9, 0.999]},
          {"type": "SGD", "momentum": 0.9}
        ],
        "batch_sizes": [32, 64]
      }
    }

}  

"""


class ModelTrainer:
    def __init__(self, model_name: str, config_file: str, model: BaseCustomModel, dataset_cls: Type[Dataset], loss_fn: Optional[nn.Module] = None):
        self.model_name = model_name
        self.config = self.load_config(config_file)[model_name]
        self.default_config = self.config["default"]
        self.combinations = self.config.get("combinations", {})
        self.model = model
        self.dataset_cls = dataset_cls
        self.loss_fn = loss_fn  # by default we rely on the config file, otherwise we adopt custom definitions.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataloaders = self.create_dataloaders()

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, "r") as file:
            return json.load(file)

    def create_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        return self.dataset_cls.get_dataloaders(batch_size=batch_size)

    def get_optimizer(self, parameters, optimizer_config: Dict[str, Any]) -> optim.Optimizer:
        if optimizer_config["type"] == "Adam":
            return optim.Adam(parameters, lr=self.current_lr, betas=optimizer_config["betas"], weight_decay=self.current_weight_decay)
        elif optimizer_config["type"] == "SGD":
            return optim.SGD(parameters, lr=self.current_lr, momentum=optimizer_config.get("momentum", 0.0), weight_decay=self.current_weight_decay)
        # here we can add support for other optimizers according to the architectures we define. 
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

    # check correctness edge cases
    def get_loss_function(self):
        if self.loss_fn is not None:
            return self.loss_fn
        loss_name = self.default_config["loss"]
        if hasattr(nn, loss_name):
            return getattr(nn, loss_name)()
        else:
            # import custom loss from custom_losses module ... to complete! 
            custom_loss_module = importlib.import_module("custom_losses")
            if hasattr(custom_loss_module, loss_name):
                return getattr(custom_loss_module, loss_name)()
            else:
                raise ValueError(f"Loss function {loss_name} not found in torch.nn or custom_losses module.")


    def train(self, epochs: int, use_combinations: bool = False, from_checkpoint: bool = False) -> None:

        if from_checkpoint:
            optimizer = self.get_optimizer(self.model.parameters(), self.default_config["optimizer"])
            start_epoch, loss, config = self.model.restart_from_checkpoint(optimizer)
            # adapt to logging libraries used in this framework.
            print(f"Resuming training from epoch {start_epoch} with loss {loss}")
        else:
            start_epoch = 0
            optimizer = None
        
        if use_combinations and self.combinations:
            combinations = list(itertools.product(
                self.combinations.get("learning_rates", [self.default_config["learning_rate"]]),
                self.combinations.get("optimizers", [self.default_config["optimizer"]]),
                self.combinations.get("batch_sizes", [self.default_config["batch_size"]])
            ))
        else:
            combinations = [(self.default_config["learning_rate"], self.default_config["optimizer"], self.default_config["batch_size"])]

        for combo in combinations:
            self.current_lr, self.current_optimizer_config, self.current_batch_size = combo
            self.current_weight_decay = self.default_config["weight_decay"] # turn it to optional 
            self.current_optimizer = self.get_optimizer(self.model.parameters(), self.current_optimizer_config)
            #self.criterion = self.loss_fn if self.loss_fn is not None else getattr(nn, self.default_config["loss"])()
            self.criterion = self.get_loss_function()

            self.dataloaders = self.create_dataloaders(self.current_batch_size)
            train_loader = self.dataloaders['train']
            val_loader = self.dataloaders['val']

            # adapt to logging libraries used in this framework.
            print(f"Training with combination: LR={self.current_lr}, Optimizer={self.current_optimizer_config}, Batch Size={self.current_batch_size}")

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0.0

                # log and save this according to libraries used in this framework. 
                # todo:  adapt to all possible input data (model specific).
                for inputs in train_loader:
                    inputs, labels = inputs["images"].to(self.device), inputs["bboxes"].to(self.device) # to edit , see TODO.
                    self.current_optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.current_optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
                
                # %5
                self.evaluate(val_loader)

                # %5
                self.model.save_checkpoint(epoch, self.current_optimizer, avg_loss)

    def evaluate(self, val_loader: DataLoader) -> None:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs in val_loader:
                inputs, labels = inputs["images"].to(self.device), inputs["bboxes"].to(self.device) # to edit , see TODO.
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        """
            plug wandb.
        """
        print(f'Validation Accuracy: {accuracy:.2f}%')



if __name__ == "__main__":

    # add classification_model_v0
    model = models.resnet18(num_classes=10) # following BaseCustomModel

    config_file = "trainer_config.json"
    dataset_cls = RefCOCOgBaseDataset()

    # optional if we don't want to use the 'default'
    custom_loss_fn = nn.CrossEntropyLoss()

    # model_name is the name in 'config.json' , it should match the naming convention of the model above. 
    trainer = ModelTrainer(model_name=model.name, config_file=config_file, model=model, dataset_cls=dataset_cls, loss_fn=custom_loss_fn)
    #trainer = ModelTrainer(model_name="Classification_v0", config_file=config_file, model=model, dataset_cls=dataset_cls, loss_fn=custom_loss_fn)
    
    trainer.train(epochs=5, use_combinations=False)
    #trainer.train(epochs=5, use_combinations=True)
    #trainer.train(epochs=5, use_combinations=False, from_checkpoint=True)