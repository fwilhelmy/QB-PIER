import time
from pytorch_lightning import Callback, Trainer, LightningModule

class EpochTimingCallback(Callback):
    def __init__(self):
        self._train_start: float = 0.0
        self._val_start:   float = 0.0

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule,
                             batch, batch_idx: int, dataloader_idx: int):
        if batch_idx == 0:
            # first batch of the epoch → mark start
            self._train_start = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        elapsed = time.time() - self._train_start
        # log under train/epoch_time at the final global_step of this epoch
        gs = trainer.global_step
        pl_module.logger.log_scalar("train/epoch_time", elapsed, step=gs)

    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule,
                                  batch, batch_idx: int, dataloader_idx: int):
        if batch_idx == 0:
            self._val_start = time.time()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        elapsed = time.time() - self._val_start
        gs = trainer.global_step
        pl_module.logger.log_scalar("val/epoch_time", elapsed, step=gs)
