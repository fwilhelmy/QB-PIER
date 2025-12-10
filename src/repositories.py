from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from LoggingService import LoggingService
from metrics import ExperimentProfiler, MetricsContainer
from models.ModelFactory import ModelFactory


class BaseRepository(ABC):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def profile(self, *args, **kwargs):
        ...

    @abstractmethod
    def save(self, *args, **kwargs):
        ...

    @abstractmethod
    def load(self, *args, **kwargs):
        ...

    def profiler(self, input_shape, epochs):
        return ExperimentProfiler(str(self.base_dir), input_shape, epochs)


class ModelRepository(BaseRepository):
    def profile(self, input_shape, epochs):
        return self.profiler(input_shape, epochs)

    def save(self, model_name: str, seed: int, model: Any, *, best: bool = False):
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

    def save_metrics(self, model_name: str, seed: int, container: MetricsContainer):
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


class AttackRepository(BaseRepository):
    def profile(self, input_shape, epochs):
        return ExperimentProfiler(str(self.base_dir), input_shape, epochs)

    def save(self, results: pd.DataFrame):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.base_dir / "results.csv", index=False)
        results.to_json(self.base_dir / "results.json", orient="records", indent=2)

    def load(self) -> pd.DataFrame:
        path = self.base_dir / "results.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def existing_pairs(self) -> Tuple[List[Dict[str, object]], set[Tuple[int, str, str]]]:
        data = self.load()
        if data.empty:
            return [], set()
        pairs = set(zip(data["seed"], data["source_model"], data["target_model"]))
        return data.to_dict("records"), pairs

    def save_predictions(
        self,
        *,
        target_model: str,
        seed: int,
        source_model: str,
        epsilon: float,
        predictions: Tensor,
        targets: Sequence[int],
    ) -> str:
        model_dir = self.base_dir / target_model
        model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / "predictions.csv"
        df = pd.DataFrame(
            {
                "seed": seed,
                "source_model": source_model,
                "target_model": target_model,
                "epsilon": epsilon,
                "sample_index": range(len(predictions)),
                "target": targets,
                "prediction": predictions.cpu().tolist(),
            }
        )
        df.to_csv(file_path, mode="a", header=not file_path.exists(), index=False)
        return str(file_path)

    def save_gradients(self, *, attacker_model: str, seed: int, gradients: List[Tuple[Tensor, Tensor]]) -> str:
        model_dir = self.base_dir / attacker_model
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"gradients_seed{seed}.pt"
        torch.save({"seed": seed, "model": attacker_model, "gradients": gradients}, path)
        return str(path)

    def save_logits(self, *, model: str, seed: int, logits: Tensor) -> str:
        model_dir = self.base_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"logits_seed{seed}.pt"
        torch.save({"seed": seed, "model": model, "logits": logits}, path)
        return str(path)
