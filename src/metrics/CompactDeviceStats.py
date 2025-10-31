# Copyright The Lightning AI team.
# Adapted from the original DeviceStatsMonitor callback:
# https://github.com/Lightning-AI/lightning

from typing import Any, Optional, Iterable, Union
import psutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CompactDeviceStats(Callback):
    """Compact callback to log basic system performance metrics.

    Logs:
      - performance/cpu_percent
      - performance/mem_used_gb
      - performance/gpu_mem_allocated_gb
    """

    def __init__(self, log_on: Union[str, Iterable[str]] = "train") -> None:
        super().__init__()
        self._log_on = {log_on} if isinstance(log_on, str) else set(log_on)

    def _log_resources(self, pl_module: "pl.LightningModule") -> None:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used_gb = (mem.total - mem.available) / (1024 ** 3)

        metrics = {
            "performance/cpu_percent": cpu_pct,
            "performance/mem_used_gb": mem_used_gb,
        }

        if torch.cuda.is_available():
            metrics["performance/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        self._log_resources(pl_module)
