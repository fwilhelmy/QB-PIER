from __future__ import annotations

import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List

import torch
from tqdm import tqdm

from BaseRepository import BaseRepository
from LoggingService import LoggingService
from metrics import MetricsContainer
from models.ModelFactory import ModelFactory
from __init__ import __version__

try:  # pragma: no cover - optional dependency during unit tests
    from pytorch_lightning import Callback, LightningModule, Trainer
except Exception:  # pragma: no cover - fallback when Lightning is unavailable
    class Callback:  # type: ignore
        pass

    class LightningModule:  # type: ignore
        pass

    class Trainer:  # type: ignore
        pass


class ModelProfiler:
    """Collect runtime metadata for each trained model and persist it to disk."""

    def __init__(self, log_dir: str, input_shape: tuple[int, ...], epochs: int) -> None:
        self.log_dir = log_dir
        self.input_shape = input_shape
        self.epochs = epochs
        self._start_time: datetime | None = None
        self._last_duration: float | None = None

    @contextmanager
    def profile(self, extra_info: Dict[str, Any] | None = None) -> Generator["ModelProfiler", None, None]:
        """Context manager for profiling arbitrary blocks (e.g., attacks)."""

        self._begin_run()
        try:
            yield self
        finally:
            self.save_experiment(extra_info=extra_info)

    def to_callback(self) -> Callback:
        """Return a Lightning callback that proxies events to this profiler."""

        return _ModelProfilerCallback(self)

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


class _ModelProfilerCallback(Callback):
    """Lightning adapter that delegates profiling to :class:`ModelProfiler`."""

    def __init__(self, profiler: ModelProfiler) -> None:
        super().__init__()
        self.profiler = profiler

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        self.profiler._begin_run()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        duration = self.profiler.save_experiment()

        save_model_info = getattr(pl_module, "save_model_info", None)
        if callable(save_model_info):
            save_model_info(duration)


class ModelRepository(BaseRepository):
    def profile(self, input_shape: tuple[int, ...], epochs: int) -> ModelProfiler:
        return ModelProfiler(str(self.base_dir), input_shape, epochs)

    def save(self, model_name: str, seed: int, model: Any, *, best: bool = False) -> Path:
        model_root = self.base_dir / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        (model_root / ".model").write_text(model_name)
        seed_dir = model_root / str(seed)
        seed_dir.mkdir(exist_ok=True)
        ckpt_dir = seed_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / ("best.ckpt" if best else "final.ckpt")
        torch.save(getattr(model, "state_dict", lambda: model)(), ckpt_path)
        return ckpt_path

    def save_metrics(self, model_name: str, seed: int, container: MetricsContainer) -> None:
        (self.base_dir / model_name / str(seed)).mkdir(parents=True, exist_ok=True)
        container.save_metrics()

    def crawl(self) -> List[Path]:
        experiments: List[Path] = []
        for root, _, files in os.walk(self.base_dir):
            if ".model" in files:
                experiments.append(Path(root))
        experiments.sort()
        return experiments

    def _best_checkpoint(self, ckpt_dir: Path) -> Path:
        ckpts = sorted([f for f in ckpt_dir.iterdir() if f.suffix == ".ckpt"], key=self._score)
        return ckpts[0]

    @staticmethod
    def _score(path: Path) -> float:
        match = re.search(r"loss=(\d+.\d+)", path.name)
        return float(match.group(1)) if match else float("inf")

    def load_models(self, path: Path, *, best: bool, map_location: str) -> Dict[int, Any]:
        model_name = (path / ".model").read_text().strip()
        models: Dict[int, Any] = {}
        for seed_dir in path.iterdir():
            if not seed_dir.is_dir() or seed_dir.name == ".hydra" or not seed_dir.name.isdigit():
                continue
            ckpt = seed_dir / "final.ckpt"
            if best:
                ckpt = self._best_checkpoint(seed_dir / "checkpoints")
            model = ModelFactory.load(model_name, checkpoint=str(ckpt), map_location=map_location, log_dir=str(seed_dir))
            models[int(seed_dir.name)] = model
        return models

    def load_metrics(self, path: Path) -> Dict[int, MetricsContainer]:
        metrics: Dict[int, MetricsContainer] = {}
        for seed_dir in sorted(path.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            container = MetricsContainer(str(seed_dir))
            container.load_metrics(str(seed_dir))
            metrics[int(seed_dir.name)] = container
        return metrics

    def load(self, *, load: str = "final", map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
        experiment: Dict[str, Dict[Any, Any]] = {}
        LoggingService(os.path.join(self.base_dir, datetime.now().strftime("%Y-%m-%d@%H-%M-%S")))
        for model_path in tqdm(self.crawl(), desc="Loading experiment"):
            name = os.path.relpath(model_path, self.base_dir)
            if load == "metrics":
                experiment[name] = self.load_metrics(model_path)
            elif load == "best":
                experiment[name] = self.load_models(model_path, best=True, map_location=map_location)
            else:
                experiment[name] = self.load_models(model_path, best=False, map_location=map_location)
        return experiment