from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import CrossEntropyLoss, MSELoss

from utils import calculate_accuracy
from Configuration import CPU_NUMBER, SAVES_DIR
import json
import os

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,  # For classification tasks
    "mse": MSELoss  # For regression tasks
}


class BaseModel(LightningModule):
    """
    Base LightningModule providing:
      - Auto-saving of constructor args via self.save_hyperparameters()
      - Configurable optimizer & loss
      - TensorBoard logging via Lightning’s logger
      - A ModelCheckpoint callback (no manual torch.save)
      - Train/val/test loops with accuracy tracking
    """

    def __init__(
        self,
        model_name: str,
        seed: str,
        log_service,
        save_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters("model_name", "seed", "save_dir")

        self.model_name: str = model_name
        self.seed: str = seed
        self.log_service = log_service

        # Create directories
        model_path = os.path.join(self.__name__,  str(self.model_name), str(seed))
        self.root_dir = os.path.join(save_dir or SAVES_DIR, model_path)
        os.makedirs(self.root_dir, exist_ok=True)
        self.log_dir = os.path.join(self.log_service.get_root_dir(), model_path)
        os.makedirs(self.log_dir, exist_ok=False)

        # TODO:  Should load metrics in scenario of resuming training
        self.all_metrics: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def configure_training(
        self,
        loss_fn: str = "cross_entropy",
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        optimizer: str = "adam",
        weight_decay: float = 1e-0,
        momentum: float = 0.9,
        devices: Optional[int] = None,
    ) -> None:
        """
        Initialize the training setup.
        Lightning’s Trainer will handle checkpoint saving automatically.
        """
        self.learning_rate = learning_rate
        self.devices = devices or CPU_NUMBER
        self.max_steps = max_steps

        # Build the optimizer
        optimizer = optimizer.lower()
        if optimizer == "adamw": self.optimizer = optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "adam": self.optimizer = optim.Adam(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "sgd": self.optimizer = optim.SGD(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "momentum": self.optimizer = optim.SGD(
                self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
            )
        else: raise ValueError(f"Unknown optimizer {optimizer}")

        # Build the loss
        loss_fn = loss_fn.lower()
        assert loss_fn in SUPPORTED_LOSS_FUNCTIONS, f"Unsupported loss fn: {loss_fn}"
        self.loss_fn = SUPPORTED_LOSS_FUNCTIONS[loss_fn]()

        self.configure_trainer()

    def configure_trainer(self) -> None:
        # Set up TensorBoard logger
        tb_logger = TensorBoardLogger(save_dir=str(self.log_dir), name='')

        # Create a ModelCheckpoint callback
        ckpt_dir = os.path.join(self.root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=False)
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )

        # Finally, create the Trainer. It will:
        self.trainer: Trainer = Trainer(
            max_steps=max_steps,
            logger=tb_logger,
            default_root_dir=self.root_dir,
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

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
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
            {
                "val_loss": loss, 
                "val_acc": acc
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False
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
            {
                "test_loss": loss, 
                "test_acc": acc
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )

        self.all_metrics["test_loss"].append(loss.item())
        self.all_metrics["test_acc"].append(acc.item())

        return loss

    def fit(
        self,
        train_dataloaders: Union[Any, list],
        val_dataloaders: Union[Any, list],
    ) -> None:
        """
        A thin wrapper for training the model using PyTorch Lightning's Trainer.
        """
        assert self.trainer is not None, "Trainer not configured. Call configure_training() first."

        self.trainer.fit(self, train_dataloaders, val_dataloaders)
        self.trainer.save_checkpoint(f"{self.root_dir}/final.ckpt")

    def save_metrics(self) -> str:
        metrics_path = os.path.join(self.root_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.all_metrics, f, indent=4)
        return metrics_path
