from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.optim import Adam, SGD
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import CrossEntropyLoss, MSELoss

from utils import calculate_accuracy
from Configuration import CPU_NUMBER
from LoggingService import LoggingService
import os
from pytorch_lightning.callbacks import ModelCheckpoint

SUPPORTED_OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD,
}

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
        devices: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.devices = devices or CPU_NUMBER
        
        self.all_metrics: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "weight_l2_norm": [],
            "learning_rate": [],
        }

        self._configure_logging()

    def _configure_logging(self) -> None:
        """Set up external LoggingService and model directory."""
        self.log_service = LoggingService()
        self.root_dir = Path(self.log_service.get_root_dir())
        self.model_dir = self.root_dir / self.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def configure_training(self,
                           loss_fn = CrossEntropyLoss,
                           learning_rate: float = 1e-3,
                           max_epochs: int = 10,
                           optimizer: str = "adam",) -> None:
        """Initialize the PyTorch Lightning Trainer."""

        assert optimizer.lower() not in SUPPORTED_OPTIMIZERS, f"Unsupported optimizer: {optimizer}"
        self.optimizer_class = SUPPORTED_OPTIMIZERS[optimizer.lower()]

        assert loss_fn.lower() not in SUPPORTED_LOSS_FUNCTIONS, f"Unsupported loss function: {loss_fn}"
        self.loss_function = SUPPORTED_LOSS_FUNCTIONS[loss_fn.lower()]()

        self.learning_rate = learning_rate

        tb_logger = TensorBoardLogger(
            save_dir=str(self.root_dir / "tensorboard"),
            name=self.model_name,
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(self.model_dir / "checkpoints"),
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )

        self.trainer: Trainer = Trainer(
            accelerator="auto",
            devices=self.devices,
            max_epochs=max_epochs,
            logger=tb_logger,
            default_root_dir=str(self.root_dir),
            callbacks=[checkpoint_cb],
        )

    def configure_optimizers(self) -> Any:
        return self.optimizer_class(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        # compute L2 norm of all weights
        l2_norm = torch.sqrt(
            sum(torch.sum(param ** 2) for param in self.parameters())
        )

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
                "weight_l2_norm": l2_norm,
                "learning_rate": self.learning_rate,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
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
        return loss

    def on_train_epoch_end(self) -> None:
        """Append logged metrics to metrics after each epoch."""
        epoch_metrics = self.trainer.callback_metrics
        for key in self.all_metrics.keys():
            if key in epoch_metrics:
                self.all_metrics[key].append(epoch_metrics[key].item())

    def fit(
        self,
        train_dataloaders: Union[Any, list],
        val_dataloaders: Union[Any, list],
    ) -> None:
        """Kick off training and save model at the end."""
        assert self.trainer is not None, "Trainer not configured. Call configure_training() first."
        self.trainer.fit(self, train_dataloaders, val_dataloaders)
        self.log_service.save_model(self)
        self.log_service.save_metrics(self)

    def load_model(logdir, model_name, map_location=None) -> nn.Module:
        """ Load the model's state dictionary """
        load_path = os.path.join(logdir, f"{model_name}.pth")
        # LOAD with pyTorch Lightning
        model = BaseModel.load_from_checkpoint(load_path, map_location=map_location)
 