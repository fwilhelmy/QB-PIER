from __future__ import annotations

import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, override

import torch
from tqdm import tqdm

from models import BaseModel
from BaseRepository import BaseRepository
from LoggingService import LoggingService
from metrics import MetricsContainer
from models.ModelFactory import ModelFactory
from __init__ import __version__

log = logging.getLogger(__name__)

class ModelRepository(BaseRepository):
    def __init__(self, input_shape: tuple[int, ...], epochs: int):
        super().__init__(base_dir=Path("models"))
        self.input_shape = input_shape
        self.epochs = epochs

    @override
    @contextmanager
    def profile(self, extra_info: Dict[str, Any] | None = None):
        start_time = datetime.now()
        try:
            yield self
        except Exception as err:
            log.exception(err)
            raise err
        finally:
            end_time = datetime.now()
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

            if extra_info:
                info.update(extra_info)

            os.makedirs(self.base_dir, exist_ok=True)
            with open(os.path.join(self.base_dir, "training.info"), "w", encoding="utf-8") as fp:
                json.dump(info, fp, indent=2)

            self._last_duration = duration
            return duration

    def save(self, model: BaseModel, seed: int):
        seed_dir = os.path.join(self.base_dir, model.log_dir, str(seed))
        os.makedirs(seed_dir, exist_ok=True)
        
        # TODO : Get the ModelFactory name of the specific BaseModel instance
        model_name = ModelFactory.name_of(model)
        (model_root / ".model").write_text(model_name) # TODO : Save model name to .model file
        model.trainer.save_checkpoint(f"{seed_dir}/final.ckpt")
        model.metrics.save_metrics()

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