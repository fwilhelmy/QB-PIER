"""Utilities for persisting per-model training metadata."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Tuple

import torch

try:  # pragma: no cover - optional dependency during unit tests
    from pytorch_lightning import Callback, LightningModule, Trainer
except Exception:  # pragma: no cover - fallback when Lightning is unavailable
    class Callback:  # type: ignore
        """Minimal stand-in used when PyTorch Lightning is unavailable."""

        pass

    class LightningModule:  # type: ignore
        pass

    class Trainer:  # type: ignore
        pass

from __init__ import __version__

log = logging.getLogger(__name__)


class ExperimentProfiler:
    """Collect runtime metadata for each trained model and persist it to disk."""

    def __init__(self, log_dir: str, input_shape: Tuple[int, ...], epochs: int) -> None:
        self.log_dir = log_dir
        self.input_shape = input_shape
        self.epochs = epochs
        self._start_time: datetime | None = None
        self._last_duration: float | None = None

    @contextmanager
    def profile(self, extra_info: Dict[str, Any] | None = None) -> Generator["ExperimentProfiler", None, None]:
        """Context manager for profiling arbitrary blocks (e.g., attacks)."""

        self._begin_run()
        try:
            yield self
        finally:
            self.save_experiment(extra_info=extra_info)

    def to_callback(self) -> Callback:
        """Return a Lightning callback that proxies events to this profiler."""

        return _ExperimentProfilerCallback(self)

    def save_experiment(self, extra_info: Dict[str, Any] | None = None) -> float:
        """Persist overall experiment metadata (training.info)."""

        info, duration = self._finalize_run()
        if extra_info:
            info.update(extra_info)

        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "training.info"), "w", encoding="utf-8") as fp:
            json.dump(info, fp, indent=2)

        self._last_duration = duration
        return duration

    def _begin_run(self) -> None:
        self._start_time = datetime.now()

    def _finalize_run(self) -> tuple[Dict[str, Any], float]:
        end_time = datetime.now()
        start_time = self._start_time or end_time
        duration = (end_time - start_time).total_seconds()

        info: Dict[str, Any] = {
            "pipeline_version": __version__,
            "cuda_available": torch.cuda.is_available(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
        }

        commit = self._current_commit()
        if commit:
            info["git_commit"] = commit

        self._start_time = None
        return info, duration

    @staticmethod
    def _current_commit() -> str | None:
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
                .decode()
                .strip()
            )
            return commit
        except Exception:  # pragma: no cover - informational best effort
            return None


class _ExperimentProfilerCallback(Callback):
    """Lightning adapter that delegates profiling to :class:`ExperimentProfiler`."""

    def __init__(self, profiler: ExperimentProfiler) -> None:
        super().__init__()
        self.profiler = profiler

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        self.profiler._begin_run()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        duration = self.profiler.save_experiment()

        save_model_info = getattr(pl_module, "save_model_info", None)
        if callable(save_model_info):
            save_model_info(duration)
