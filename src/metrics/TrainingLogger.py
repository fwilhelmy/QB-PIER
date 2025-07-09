from pytorch_lightning.callbacks.callback import Callback
from collections import defaultdict
import torch

class TrainingLogger(Callback):
    def __init__(self, ratio: float = 1.0):
        super().__init__()
        assert 0 < ratio <= 1.0
        self.ratio = ratio
        self._train_buf = defaultdict(list)
        self._val_buf   = defaultdict(list)
        self._pending_thresholds = []

    def on_train_epoch_start(self, trainer, pl_module):
        total = trainer.num_training_batches
        thresholds = []
        r = self.ratio
        while r <= 1.0:
            thresholds.append(r)
            r += self.ratio
        if 1.0 not in thresholds:
            thresholds.append(1.0)
        self._pending_thresholds = [int(t * total) for t in thresholds]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        for k, v in outputs.items():
            self._train_buf[k].append(v.item() if torch.is_tensor(v) else v)

        if self._pending_thresholds and batch_idx + 1 >= self._pending_thresholds[0]:
            self._pending_thresholds.pop(0)
            avg = {k: sum(vs) / len(vs) for k, vs in self._train_buf.items()}
            pl_module.metrics.log_dict("train", avg, step)
            self._train_buf.clear()

        pl_module.log_dict({f"train/{k}": vs for k, vs in outputs.items()})

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
