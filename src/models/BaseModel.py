from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import json
import pandas as pd

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from lightning.pytorch.loggers.logger import Logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from utils import calculate_accuracy
from Configuration import CPU_NUMBER
from src.logging.LoggingService import LoggingService
from logging import TrainingLogger, MetricsContainer

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
        self.metrics = MetricsContainer(self.log_dir)

    def configure_directories(self) -> None:
        # Build folder paths using class name, model_name, seed
        relative_path = os.path.join(self.__class__.__name__, str(self.model_name), str(self.seed))
        self.log_dir = os.path.join(LoggingService().get_root_dir(), relative_path)
        self._init_directories()

    @rank_zero_only
    def _init_directories(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        metrics_dir = os.path.join(self.log_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

    def configure_training(
        self,
        loss_fn: str = "cross_entropy",
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        optimizer: str = "adam",
        weight_decay: float = 1e-0,
        momentum: float = 0.9,
        devices: Optional[int] = None,
        train_log_interval: float = 0.5,
        eval_log_interval: float = 0.5
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
        self.hparams["train_log_interval"]  = train_log_interval
        self.hparams["eval_log_interval"]   = eval_log_interval

        self._init_trainer()

    def _init_trainer(self) -> None:
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
        tb_logger = TensorBoardLogger(save_dir=self.log_dir, name=None, version="tensorboard")
        tr_logger = TrainingLogger(ratio=self.hparams["train_log_interval"])
        
        # Checkpoint callback
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch={epoch:02d}-loss={val/loss:.4f}",
            auto_insert_metric_name=False,
            save_top_k=3,
            save_on_train_epoch_end=False,
            monitor="val/loss",
            mode="min"
        )

        # Build the Trainer
        self.trainer: Trainer = Trainer(
            max_steps=self.hparams["max_steps"],
            logger=[tb_logger],
            default_root_dir=self.log_dir,
            callbacks=[checkpoint_cb, tr_logger],
            val_check_interval=self.hparams["eval_log_interval"],
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['log_dir'] = self.log_dir

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model_name = self.hparams["model_name"]
        self.seed = self.hparams["seed"]

        self.configure_directories()
        self._init_trainer()
        self.metrics.load(checkpoint['log_dir'])
        
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError # must be implemented by each subclass
    
    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)
        
        params_l2 = sum((p.norm(2).item() ** 2) for p in self.parameters() if p.requires_grad)
        params_l2 = params_l2 ** 0.5

        return {"loss": loss, "acc": acc, "weights_l2": params_l2}

    def on_train_epoch_end(self):
        self.metrics.log_epoch(self.current_epoch, self.global_step)

        with torch.no_grad():
            grad_l2 = {name: param.grad.data.norm(2).item()
                     for name, param in self.named_parameters()
                     if param.grad is not None}
            self.metrics.log_dict("grad_l2", grad_l2)

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        return {"loss": loss, "acc": acc}

    def fit(
        self,
        train_dataloaders: Union[Any, list],
        val_dataloaders: Union[Any, list],
    ) -> None:
        """
        Thin wrapper that calls Lightning’s Trainer.fit(), then saves a final checkpoint and metrics.
        """
        with self.metrics.timeit("training_time", step=self.global_step):
            assert hasattr(self, "trainer"), "Trainer not configured. Call configure_training() first."
            self.trainer.fit(self, train_dataloaders, val_dataloaders)
            self.trainer.save_checkpoint(f"{self.log_dir}/final.ckpt")
            self.metrics.save()