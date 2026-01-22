from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List
from models.ModelFactory import ModelFactory

def crawl_experiments(path: str) -> List[str]:
    return [str(path) for path in _crawl_experiment_paths(path)]


def _crawl_experiment_paths(path: str | Path) -> List[Path]:
    experiments: List[Path] = []
    for root, _, files in os.walk(path):
        if ".model" in files:
            experiments.append(Path(root))
    experiments.sort()
    return experiments


def _score_checkpoint(path: Path) -> float:
    match = re.search(r"loss=(\d+\.\d+)", path.name)
    return float(match.group(1)) if match else float("inf")


def _best_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted([f for f in checkpoint_dir.iterdir() if f.suffix == ".ckpt"], key=_score_checkpoint)
    return ckpts[0] if ckpts else None


def _select_checkpoint(seed_dir: Path, load: str) -> Path | None:
    final_ckpt = seed_dir / "final.ckpt"
    if load == "best":
        best_ckpt = _best_checkpoint(seed_dir / "checkpoints")
        return best_ckpt or (final_ckpt if final_ckpt.exists() else None)
    if final_ckpt.exists():
        return final_ckpt
    return _best_checkpoint(seed_dir / "checkpoints")


def load_experiment(path: str, *, load: str = "final", map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
    base_dir = Path(path)
    experiments: Dict[str, Dict[Any, Any]] = {}
    for model_path in crawl_experiments(path):
        model_path = Path(model_path)
        model_name = (model_path / ".model").read_text().strip()
        experiment_name = os.path.relpath(model_path, base_dir)
        models: Dict[int, Any] = {}
        for seed_dir in model_path.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            checkpoint = _select_checkpoint(seed_dir, load)
            if checkpoint is None:
                continue
            model = ModelFactory.load(
                model_name,
                checkpoint=str(checkpoint),
                map_location=map_location,
                log_dir=str(seed_dir),
            )
            models[int(seed_dir.name)] = model
        if models:
            experiments[experiment_name] = models
    return experiments
