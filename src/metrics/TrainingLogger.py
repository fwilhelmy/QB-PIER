from pytorch_lightning.callbacks.callback import Callback
from collections import defaultdict
import torch

class TrainingLogger(Callback):
    def __init__(self):
        super().__init__()
        self._train_buf = defaultdict(list)
        self._val_buf   = defaultdict(list)

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_buf.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for k, v in outputs.items():
            self._train_buf[k].append(v.item() if torch.is_tensor(v) else v)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._train_buf:
            return
        step = trainer.global_step
        avg = {k: sum(vs) / len(vs) for k, vs in self._train_buf.items()}
        pl_module.metrics.log_dict("train", avg, step)
        for k, v in avg.items():
            pl_module.log(f"train/{k}", v)
        self._train_buf.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # now correctly buffer on the callback
        for k, v in outputs.items():
            self._val_buf[k].append(v.item() if torch.is_tensor(v) else v)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._val_buf:
            return
        step = trainer.global_step
        avg = {k: sum(vs) / len(vs) for k, vs in self._val_buf.items()}
        pl_module.metrics.log_dict("val", avg, step=step)
        for k, v in avg.items():
            pl_module.log(f"val/{k}", v)
        self._val_buf.clear()
