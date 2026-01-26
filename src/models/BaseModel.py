import logging
import os
from typing import Any, Dict, Tuple, Union

from numpy import gradient
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler as lr_schedulers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from utils import calculate_accuracy
from metrics import MetricsContainer, CompactDeviceStats, GradientProfiler

log = logging.getLogger(__name__)

AVAILABLE_OPTIMISERS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}

AVAILABLE_SCHEDULERS = {
    "cosineannealinglr": lr_schedulers.CosineAnnealingLR,
    "exponentiallr": lr_schedulers.ExponentialLR,
    "multisteplr": lr_schedulers.MultiStepLR,
    "reducelronplateau": lr_schedulers.ReduceLROnPlateau,
    "steplr": lr_schedulers.StepLR,
}

class BaseModel(LightningModule):
    """
    Base Model class for PyTorch Lightning models.
    """

    def __init__(
        self,
        log_dir: str,
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
        self.save_hyperparameters("log_dir", "input_size", "output_size")
        self.log_dir = log_dir
        self.input_size = input_size
        self.output_size = output_size
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
            'log_interval': config.get("log_interval", 1.0),
            'lr_scheduler': config.get("lr_scheduler"),
            'fast_dev_run': config.get("fast_dev_run", False),
        })

        self._init_training()

    def _init_training(self) -> None:
        " Initializes the PyTorch Lightning Trainer with the specified hyperparameters, loggers and callbacks."
        # Configure loss function
        self.configure_loss_fn()

        # Create loggers
        tb_logger = TensorBoardLogger(save_dir=self.log_dir, name=None, version="tensorboard")

        profiler_dirpath = os.path.join(self.log_dir, "profiler")
        self.profiler = PyTorchProfiler(
            dirpath=profiler_dirpath,
            filename="trace",
            export_to_chrome=True,
            record_module_names=True,
            row_limit=1000,
            sort_by_key="cpu_time"
        )
        
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
        device_monitor = CompactDeviceStats()
        gradient_profiler = GradientProfiler()

        # Build the Trainer
        self.trainer: Trainer = Trainer(
            max_epochs=self.hparams["max_epochs"],
            accelerator=self.hparams["accelerator"],
            devices=self.hparams["devices"],
            logger=[tb_logger, self.metrics],
            default_root_dir=self.log_dir,
            callbacks=[checkpoint_cb, device_monitor, gradient_profiler],
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
            log.warning("Could not load metrics. No directory provided in checkpoint.")

    def configure_loss_fn(self) -> None:
        self.loss_fn = CrossEntropyLoss()

    def configure_optimizers(self) -> Any:
        name = self.hparams["optimizer"].lower()
        optimiser_cls = AVAILABLE_OPTIMISERS.get(name)
        if optimiser_cls is None:
            available = ", ".join(sorted(AVAILABLE_OPTIMISERS))
            raise ValueError(f"Unsupported optimizer: {name}. Available: {available}")

        optimiser_kwargs = dict(self.hparams.get("optimiser_config", {}))
        optimizer = optimiser_cls(self.parameters(), lr=self.hparams["learning_rate"], **optimiser_kwargs)

        scheduler_cfg = self.hparams.get("lr_scheduler")
        if not scheduler_cfg:
            return optimizer

        cfg = dict(scheduler_cfg)
        scheduler_name = cfg.pop("name", None)
        scheduler_cls = AVAILABLE_SCHEDULERS.get(scheduler_name.lower())
        if scheduler_cls is None:
            available = ", ".join(sorted(AVAILABLE_SCHEDULERS))
            raise ValueError(f"Unsupported scheduler: {scheduler_name}. Available: {available}")

        interval = cfg.pop("interval", "epoch")
        frequency = cfg.pop("frequency", 1)
        monitor = cfg.pop("monitor", None)
        strict = cfg.pop("strict", False)

        scheduler = scheduler_cls(optimizer, **cfg)

        scheduler_config: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
            "strict": strict,
        }

        if monitor is not None:
            scheduler_config["monitor"] = monitor
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler_config["monitor"] = "val/loss"

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError # must be implemented by each subclass

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)
        
        weights_l2 = sum((p.norm(2).item() ** 2) for p in self.parameters() if p.requires_grad)
        weights_l2 = weights_l2 ** 0.5

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/weights_l2", weights_l2, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "acc": acc, "weights_l2": weights_l2}

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = calculate_accuracy(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "acc": acc}

    def on_train_epoch_end(self):
        self.metrics.log_epoch(self.global_step)

    def save_model_info(self, duration: float | None = None) -> None:
        """Persist model architecture metadata (model.info)."""

        details: list[str] = []

        if summary is not None:
            try:
                model_summary = str(
                    summary(
                        self,
                        input_data=tuple(self.input_size),
                        col_names=("num_params", "input_size", "output_size"),
                        verbose=0,
                    )
                )
                details.append(model_summary)
            except Exception as exc:  # pragma: no cover - only triggered when summary fails
                log.warning("Failed to generate model summary: %s", exc)
        else:  # pragma: no cover - summary dependency missing during some tests
            log.warning("torchsummary is not installed; skipping model summary generation")

        epochs = int(self.hparams.get("max_epochs", 0))
        total_duration = duration if duration is not None else 0.0
        details.append(f"Trained epochs: {epochs}")
        if epochs:
            details.append(f"Training time per epoch: {total_duration/epochs:.2f} seconds")
        else:
            details.append("Training time per epoch: undefined")
        details.append(f"Total training time (per seed): {total_duration:.2f} seconds")

        info = "\n".join(details)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "model.info"), "w", encoding="utf-8") as fp:
            fp.write(info)

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
        try:
            self.trainer.fit(self, train_dataloaders, val_dataloaders)
        except Exception as err:
            log.exception(err)
            raise err
        finally:
            # self.profiler.teardown(stage=None)
            self.metrics.save_metrics()
            self.trainer.save_checkpoint(f"{self.log_dir}/final.ckpt")