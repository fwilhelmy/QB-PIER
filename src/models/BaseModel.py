from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import json
import pandas as pd

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import calculate_accuracy
from Configuration import CPU_NUMBER
from LoggingService import LoggingService

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,  # For classification tasks
    "mse":           MSELoss             # For regression tasks
}

class BaseModel(LightningModule):
    """
    Base Model class for PyTorch Lightning models.
    """

    def __init__(
        self,
        model_name: str,
        seed: int
    ):
        super().__init__()
        self.save_hyperparameters("model_name", "seed")
        self.model_name = model_name
        self.seed = seed

        self.configure_directories()

        # in‐memory nested metrics
        self.all_metrics = {
            "train": {"loss": [], "acc": [], "step": [], "epoch": []},
            "val":   {"loss": [], "acc": [], "step": [], "epoch": []},
        }

    def configure_directories(self) -> None:
        # Build folder paths using class name, model_name, seed
        relative_path = os.path.join(self.__class__.__name__, str(self.model_name), str(self.seed))
        self.log_dir = os.path.join(LoggingService().get_root_dir(), relative_path)
        os.makedirs(self.log_dir, exist_ok=True)

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
        Record each hyperparameter into self.hparams so that Lightning
        serializes them into the checkpoint.
        """
        self.hparams["loss_fn"]       = loss_fn.lower()
        self.hparams["learning_rate"] = learning_rate
        self.hparams["max_steps"]     = max_steps
        self.hparams["optimizer"]     = optimizer.lower()
        self.hparams["weight_decay"]  = weight_decay
        self.hparams["momentum"]      = momentum
        self.hparams["devices"]       = devices or CPU_NUMBER

        self._instantiate_from_hparams()

    def _instantiate_from_hparams(self) -> None:
        """
        1) Build loss function
        2) Create loggers
        3) Create checkpoint callback
        4) Instantiate Trainer
        """
        # Build Loss Function
        lf = self.hparams["loss_fn"]
        if lf not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss_fn: {lf}")
        self.loss_fn = SUPPORTED_LOSS_FUNCTIONS[lf]()

        # Create loggers
        tb_logger  = TensorBoardLogger(save_dir=self.log_dir, name="", version="tensorboard")
        csv_logger = CSVLogger(save_dir=self.log_dir, name="", version="metrics")

        # Checkpoint callback
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min"
        )

        # Build the Trainer
        self.trainer: Trainer = Trainer(
            max_steps=self.hparams["max_steps"],
            logger=[tb_logger, csv_logger],
            default_root_dir=self.log_dir,
            callbacks=[checkpoint_cb],
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['log_dir'] = self.log_dir

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model_name = self.hparams["model_name"]
        self.seed = self.hparams["seed"]

        # reload metrics.json if present
        metrics_file = os.path.join(checkpoint['log_dir'], "metrics/full_metrics.parquet")
        if os.path.exists(metrics_file):
            # load existing nested structure
            df = pd.read_parquet(metrics_file)
            self.all_metrics = df.to_dict(orient="list")
        else:
            # fresh nested structure
            self.all_metrics = {
                "train": {"loss": [], "acc": [], "step": [], "epoch": []},
                "val":   {"loss": [], "acc": [], "step": [], "epoch": []},
            }

        self.configure_directories()

        self._instantiate_from_hparams()

    def configure_optimizers(self) -> Any:
        if self.hparams["optimizer"] == "adamw":
            return optim.AdamW(
                self.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == "adam":
            return optim.Adam(
                self.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == "sgd":
            return optim.SGD(
                self.parameters(),
                lr=self.hparams["learning_rate"],
                weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == "momentum":
            return optim.SGD(
                self.parameters(),
                lr=self.hparams["learning_rate"],
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams["weight_decay"]
            )
        else: raise ValueError(f"Unknown optimizer: {self.hparams['optimizer']}")

    def _log_metrics(
        self,
        phase: str,
        metrics: Dict[str, float],
        prog_bar: bool = False
        ) -> None:
        """
        Emits all metrics to Lightning (TensorBoard & CSV) with keys like "train/loss",
        """
        flat = {f"{phase}/{k}": v for k, v in metrics.items()} # Flattened keys
        self.log_dict(flat, on_step=False, on_epoch=True, prog_bar=prog_bar)

        for k, v in metrics.items():
            if k not in self.all_metrics[phase]:
                raise KeyError(f"Unknown metric for {phase}: {k}")
            self.all_metrics[phase][k].append(v)

        self.all_metrics[phase]["step"].append(self.global_step)
        self.all_metrics[phase]["epoch"].append(self.current_epoch)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError # must be implemented by each subclass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self._log_metrics("train", {"loss": loss.item(), "acc": acc.item()})
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self._log_metrics("val", {"loss": loss.item(), "acc": acc.item()})
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def fit(
        self,
        train_dataloaders: Union[Any, list],
        val_dataloaders: Union[Any, list],
    ) -> None:
        """
        Thin wrapper that calls Lightning’s Trainer.fit(), then saves a final checkpoint and metrics.
        """
        assert hasattr(self, "trainer"), "Trainer not configured. Call configure_training() first."
        self.trainer.fit(self, train_dataloaders, val_dataloaders)
        self.trainer.save_checkpoint(f"{self.log_dir}/final.ckpt")
        self.save_metrics()

    def save_metrics(self) -> str:
        """
        Dump the in-memory `all_metrics` to parquet. Can be reloaded in on_load_checkpoint().
        """
        metrics_path = os.path.join(self.log_dir, "metrics/full_metrics.parquet")
        df = pd.DataFrame(self.all_metrics)
        df.to_parquet(metrics_path)
        return metrics_path
