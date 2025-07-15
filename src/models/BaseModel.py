from typing import Any, Dict, Tuple, Union
import os
from hydra.core.hydra_config import HydraConfig

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from .OptimiserFactory import OptimiserFactory

from utils import calculate_accuracy
from metrics import TrainingLogger, MetricsContainer

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
        seed: int,
        input_size: Tuple[int, int, int],
        output_size: int,
    ):
        """Initialize the base model.

        Parameters
        ----------
        model_name: str
            Name identifying the model instance.
        seed: int
            Random seed used for the experiment.
        input_size: Any
            Shape or dimensionality of the model input.
        output_size: Any
            Number of outputs produced by the model.
        """
        super().__init__()
        self.save_hyperparameters("model_name", "seed", "input_size", "output_size")
        self.model_name = model_name
        self.seed = seed
        self.input_size = input_size
        self.output_size = output_size

        self.configure_directories()
        self.metrics = MetricsContainer(self.log_dir)

    def configure_directories(self) -> None:
        # Build folder paths using class name, model_name, seed
        self.log_dir = os.path.join(HydraConfig.get().run.dir, f"seed={self.seed}")
        self._init_directories()

    @rank_zero_only
    def _init_directories(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def configure_training(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.hparams.update({
            'loss_fn': config.get("loss_fn", "cross_entropy").lower(),
            'optimizer': config.get("optimizer", "adam").lower(),
            'max_steps': config.get("max_steps", 1000),
            'learning_rate': config.get("learning_rate", 1e-3),
            'optimiser_config': config.get("optimiser_config", {}),
            'accelerator': config.get("accelerator", "auto"),
            'devices': config.get("devices", 1),
            'train_log_interval': config.get("train_log_interval", 0.5),
            'eval_log_interval': config.get("eval_log_interval", 0.5)
        })

        self._init_training()

    def _init_training(self) -> None:
        " Initializes the PyTorch Lightning Trainer with the specified hyperparameters, loggers and callbacks."
        # Loss function
        loss_fn = self.hparams["loss_fn"]
        if loss_fn not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss_fn: {loss_fn}")
        self.loss_fn = SUPPORTED_LOSS_FUNCTIONS[loss_fn]()

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
            accelerator=self.hparams["accelerator"],
            devices=self.hparams["devices"],
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
        self.configure_training(checkpoint['hyper_parameters'])
        self.metrics.load_metrics(checkpoint['log_dir'])
    
    def configure_optimizers(self) -> Any:
        """Create the optimiser using :class:`OptimiserFactory`."""

        return OptimiserFactory(
            self.hparams["optimizer"],
            self.parameters(),
            self.hparams["learning_rate"],
            **self.hparams.get("optimiser_config", {}),
        )

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
        self.metrics.log_epoch(self.global_step)

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
            self.metrics.save_metrics()