import logging
import os
from typing import Any, Dict, Tuple, Union

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import CrossEntropyLoss

from metrics import LoggingService, MetricsContainer, TrainingLogger
from utils import calculate_accuracy

log = logging.getLogger(__name__)

class BaseModel(LightningModule):
    """
    Base Model class for PyTorch Lightning models.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int, int],
        output_size: int,
        **kwargs
    ):
        """Initialize the base model.

        Parameters
        ----------
        input_size: Any
            Shape or dimensionality of the model input.
        output_size: Any
            Number of outputs produced by the model.
        """
        super().__init__()
        self.save_hyperparameters("model_path", "input_size", "output_size")
        self.model_path = model_path
        self.input_size = input_size
        self.output_size = output_size
        self.log_dir = os.path.join(LoggingService().root_dir, self.model_path)
        self.metrics = MetricsContainer(self.log_dir)

    @rank_zero_only
    def _init_directories(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def configure_training(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.hparams.update({
            'optimizer': config.get("optimizer", "adam").lower(),
            'max_epochs': config.get("max_epochs", 1),
            'learning_rate': config.get("learning_rate", 1e-3),
            'optimiser_config': config.get("optimiser_config", {}),
            'accelerator': config.get("accelerator", "auto"),
            'devices': config.get("devices", 1),
            'fast_dev_run': config.get("fast_dev_run", False),
        })

        self._init_training()

    def _init_training(self) -> None:
        " Initializes the PyTorch Lightning Trainer with the specified hyperparameters, loggers and callbacks."
        # Configure loss function
        self.configure_loss_fn()

        # Create loggers
        tb_logger = TensorBoardLogger(save_dir=self.log_dir, name=None, version="tensorboard")
        tr_logger = TrainingLogger()

        # profiler_dirpath = os.path.join(self.log_dir, "profiler")
        # self.profiler = PyTorchProfiler(
        #     dirpath=profiler_dirpath,        # where to save files
        #     filename="trace",               # base name
        #     export_to_chrome=True,          # export .json file for Chrome Trace
        #     record_module_names=True,
        #     row_limit=1000,                 # or whatever you need
        #     # optionally you can pass profiler kwargs for torch.profiler.profile
        # )
        
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

        # Device stats monitor callback (logs hardware utilization)
        device_monitor = DeviceStatsMonitor()

        # Build the Trainer
        self.trainer: Trainer = Trainer(
            max_epochs=self.hparams["max_epochs"],
            accelerator=self.hparams["accelerator"],
            devices=self.hparams["devices"],
            logger=[tb_logger],
            default_root_dir=self.log_dir,
            callbacks=[checkpoint_cb, tr_logger, device_monitor],
            fast_dev_run=self.hparams["fast_dev_run"],
            enable_model_summary=False,
            # profiler=self.profiler,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['log_dir'] = self.log_dir

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.hparams.update(checkpoint['hyper_parameters'])
        self.configure_loss_fn()
        if 'log_dir' in checkpoint['hyper_parameters']:
            self.metrics.load_metrics(checkpoint['hyper_parameters']['log_dir'])
        else:
            log.error("No log directory provided in checkpoint.")

    def configure_loss_fn(self) -> None:
        self.loss_fn = CrossEntropyLoss()

    def configure_optimizers(self) -> Any:
        name = self.hparams["optimizer"].lower()
        if name == "adamw":
            return optim.AdamW(self.parameters(), lr=self.hparams["learning_rate"], **self.hparams.get("optimiser_config", {}))
        if name == "adam":
            return optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], **self.hparams.get("optimiser_config", {}))
        if name == "sgd":
            return optim.SGD(self.parameters(), lr=self.hparams["learning_rate"], **self.hparams.get("optimiser_config", {}))
        raise ValueError(f"Unsupported optimizer: {name}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError # must be implemented by each subclass

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)
        
        weights_l2 = sum((p.norm(2).item() ** 2) for p in self.parameters() if p.requires_grad)
        weights_l2 = weights_l2 ** 0.5

        return {"loss": loss, "acc": acc, "weights_l2": weights_l2}

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
    
    def _gradient_norms(self) -> Dict[str, float]:
        return {name: param.grad.data.norm(2).item()
            for name, param in self.named_parameters() if param.grad is not None}

    def on_train_epoch_end(self):
        self.metrics.log_epoch(self.global_step)
        # Log last computed gradient norms captured during backprop. Gradients
        # are typically cleared by the time we reach epoch end, so we cache
        # them in on_after_backward.
        if hasattr(self, "_last_grad_l2") and self._last_grad_l2:
            self.metrics.log_dict("grad_l2", self._last_grad_l2, step=self.global_step)
        else:
            # Fallback: compute if available (may be empty at epoch end)
            grad_snapshot = self._gradient_norms()
            if grad_snapshot:
                self.metrics.log_dict("grad_l2", grad_snapshot, step=self.global_step)

    def on_after_backward(self) -> None:
        # Cache gradient L2 norms right after backward, before optimizer.zero_grad()
        self._last_grad_l2 = self._gradient_norms()

    def fit(
        self,
        train_dataloaders: Union[Any, list],
        val_dataloaders: Union[Any, list],
    ) -> None:
        """
        Thin wrapper that calls Lightning’s Trainer.fit(), then saves a final checkpoint and metrics.
        """
        if not hasattr(self, "trainer"):
            raise ValueError("Trainer not configured. Call configure_training() first.")
        with self.metrics.timeit("training_time", step=self.global_step):
            try:
                self.trainer.fit(self, train_dataloaders, val_dataloaders)
            except Exception as err:
                log.exception(RuntimeError(f"Error during training: {err}"))
            finally:
                # self.profiler.teardown(stage=None)
                self.metrics.save_metrics()
                self.trainer.save_checkpoint(f"{self.log_dir}/final.ckpt")
