from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss, MSELoss

from utils import calculate_accuracy
from Configuration import CPU_NUMBER
from LoggingService import LoggingService
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import load, save
import json


SUPPORTED_OPTIMIZERS = ['adamw', 'adam', 'sgd', 'momentum']

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss, # For classification tasks
    "mse": MSELoss # For regression tasks
}

class BaseModel(LightningModule):
    """
    Base LightningModule providing:
    - Configurable optimizer & loss
    - TensorBoard logging via LoggingService
    - Train/val/test loops with accuracy tracking
    - L2‐norm of model weights logged every epoch
    """

    def __init__(
        self,
        model_name: str,
        rootdir: str
    ):
        super().__init__()

        self.all_metrics: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            #"weight_l2_norm": [],
            "learning_rate": [],
        }

        self.model_name = model_name
        self.rootdir = rootdir
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Set up external LoggingService and model directory."""
        self.log_service = LoggingService()
        # different than self.rootdir, this is the root directory for the logs, need to change names
        self.logdir = Path(self.log_service.get_root_dir())
        self.model_dir = self.logdir / self.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Todo: need to fix this, not everything is loaded now...
        # model_sate = os.path.join(self.rootdir, "model_state.pt")
        # if os.path.exists(model_sate):
        #     print(f"Model {self.model_name} already exists in {self.rootdir}, loading existing model.")
        #     # load existing model directory
        #     self.load_model()
        #     return
        
        os.makedirs(self.rootdir, exist_ok=True)

        filepath = os.path.join(self.rootdir, "configuration.txt")
        model_config = f"Model name:{self.model_name} \n{str(self)}"
        with open(filepath, "w") as file:
            file.write(model_config)

    def configure_training(self,
                           loss_fn = 'cross_entropy',
                           learning_rate: float = 1e-3,
                           max_steps: int = 1000,
                           optimizer: str = "adam",
                           weight_decay: float = 1e-0,
                           momentum: float = 0.9,
                           devices: Optional[int] = None,) -> None:
        """Initialize the PyTorch Lightning Trainer."""
        self.devices = devices or CPU_NUMBER
        optimizer = optimizer.lower()
        if optimizer == "adamw": self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adam": self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "sgd": self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "momentum": self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else: raise ValueError("Unknown optimizer {0}".format(optimizer))

        loss_fn = loss_fn.lower()
        assert loss_fn in SUPPORTED_LOSS_FUNCTIONS, f"Unsupported loss function: {loss_fn}"
        self.loss_fn = SUPPORTED_LOSS_FUNCTIONS[loss_fn.lower()]()

        self.learning_rate = learning_rate

        tb_logger = TensorBoardLogger(
            save_dir=str(self.logdir / "tensorboard"),
            name=self.model_name,
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(Path(self.rootdir) / "checkpoints"),
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
        self.max_steps = max_steps
        self.trainer: Trainer = Trainer(
            #accelerator="auto",
            #devices=self.devices,
            max_steps=max_steps,
            logger=tb_logger,
            default_root_dir=self.rootdir,
            callbacks=[checkpoint_cb],
        )

    def configure_optimizers(self) -> Any:
        return self.optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        # compute L2 norm of all weights
        #l2_norm = torch.sqrt(torch.stack([torch.sum(param ** 2) for param in self.parameters()]))

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
                #"weight_l2_norm": l2_norm,
                "learning_rate": self.learning_rate,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        self.all_metrics["train_loss"].append(loss.item())
        self.all_metrics["train_acc"].append(acc.item())
        self.all_metrics["learning_rate"].append(self.learning_rate)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        self.all_metrics["val_loss"].append(loss.item())
        self.all_metrics["val_acc"].append(acc.item())

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict(
            {"test_loss": loss, "test_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )


        self.all_metrics["test_loss"].append(loss.item())
        self.all_metrics["test_acc"].append(acc.item())
        return loss

    # def on_train_epoch_end(self) -> None:
    #     """Append logged metrics to metrics after each epoch."""
    #     epoch_metrics = self.trainer.callback_metrics
    #     for key in self.all_metrics.keys():
    #         if key in epoch_metrics:
    #             self.all_metrics[key].append(epoch_metrics[key].item())

    def save_model(self):
        filepath = os.path.join(self.rootdir, "model_state.pt")
        save(self, filepath)

    def save_metrics(self):
        with open(f"{self.rootdir}/metrics.json", "w") as f:
            json.dump(self.all_metrics, f, indent=4)

    def load_model(self, map_location=None):
        """ Load the model's state dictionary """
        filepath = os.path.join(self.rootdir, "model_state.pt")
        self = load(filepath, map_location=map_location)
        if map_location is not None: self.devices = map_location
 
    def fit(
        self,
        train_dataloaders: Union[Any, list],
        val_dataloaders: Union[Any, list],
    ) -> None:
        """Kick off training and save model at the end."""
        assert self.trainer is not None, "Trainer not configured. Call configure_training() first."
        self.trainer.fit(self, train_dataloaders, val_dataloaders)
        self.save_model()
        self.save_metrics()
        # Also should save the training figure and a log of the training in json to see how the training went
        # In a dataset when it's loaded for the first time, should save dataset figures and store into
