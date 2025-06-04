from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import json

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import calculate_accuracy
from Configuration import CPU_NUMBER, SAVES_DIR
from LoggingService import LoggingService

SUPPORTED_LOSS_FUNCTIONS = {
    "cross_entropy": CrossEntropyLoss,  # For classification tasks
    "mse": MSELoss                       # For regression tasks
}


class BaseModel(LightningModule):
    """
    Base Model class for PyTorch Lightning models.
    """

    def __init__(
        self,
        model_name: str,
        seed: int,
        save_dir: str = SAVES_DIR,
    ):
        super().__init__()
        self.save_hyperparameters("model_name", "seed", "save_dir")
        self.model_name = model_name
        self.seed = seed

        self.configure_directories()

        # Initialize in-memory metrics (Lightning does not checkpoint these)
        self.all_metrics: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }
    
    def configure_directories(self) -> None:
        """
        Configure directories for saving model checkpoints and logs.\n
        This method creates a structured directory layout based on the model name and seed.
        """
        # Build folder paths using class name, model_name, seed
        relative_path = os.path.join(self.__class__.__name__, str(self.model_name), str(self.seed))
        self.root_dir = os.path.join(self.hparams['save_dir'], relative_path)
        os.makedirs(self.root_dir, exist_ok=True)
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
        Given that self.hparams now contains:
          loss_fn, learning_rate, max_steps, optimizer, weight_decay, momentum, devices
        this method will:
          1) Build the Loss Function (CrossEntropyLoss or MSELoss)
          3) Create a new TensorBoardLogger pointed at log_dir
          4) Create a ModelCheckpoint callback pointed at root_dir/checkpoints
          5) Instantiate the Trainer
        """
        # Build Loss Function
        if self.hparams["loss_fn"] not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss_fn: {self.hparams['loss_fn']}")
        self.loss_fn = SUPPORTED_LOSS_FUNCTIONS[self.hparams["loss_fn"]]()

        # Create TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name=""
        )

        # ModelCheckpoint callback (saves .ckpt under root_dir/checkpoints)
        ckpt_dir = os.path.join(self.root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        )

        # Finally, build the Trainer
        self.trainer: Trainer = Trainer(
            max_steps=self.hparams["max_steps"],
            logger=tb_logger,
            default_root_dir=self.root_dir,
            callbacks=[checkpoint_cb]
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Lightning has already:
          • Restored model weights into self.state_dict()
          • Populated self.hparams from the checkpoint

        Now we:
          1) Configure directories based on the restored hparams
          2) Reload training metrics from disk (optional)
          3) Initialise our training instance
        """
        self.model_name = self.hparams["model_name"]
        self.seed = self.hparams["seed"]

        self.configure_directories()

        # Reload previously saved metrics into memory
        metrics_file = os.path.join(self.root_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                self.all_metrics = json.load(f)
        else: # Else, fresh, empty metrics
            self.all_metrics = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "learning_rate": [],
            }

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # must be implemented by each subclass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        # Log to Lightning so ModelCheckpoint sees val_loss later
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
                "learning_rate": self.hparams["learning_rate"]
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )

        # Keep your own in-memory history (not checkpointed):
        self.all_metrics["train_loss"].append(loss.item())
        self.all_metrics["train_acc"].append(acc.item())
        self.all_metrics["learning_rate"].append(self.hparams["learning_rate"])

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
        Thin wrapper that calls Lightning’s Trainer.fit(). After training,
        we write out a final checkpoint for convenience (optional).
        """
        assert self.trainer is not None, "Trainer not configured. Call configure_training() first."
        self.trainer.fit(self, train_dataloaders, val_dataloaders)
        self.trainer.save_checkpoint(f"{self.root_dir}/final.ckpt")
        self.save_metrics()

    def save_metrics(self) -> str:
        """
        Dump the in-memory `all_metrics` to JSON. If you want to restore
        these metrics upon next load, you must manually reload them in on_load_checkpoint().
        """
        metrics_path = os.path.join(self.root_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.all_metrics, f, indent=4)
        return metrics_path
