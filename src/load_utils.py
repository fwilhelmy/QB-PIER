from __future__ import annotations

import os
import re
from pathlib import Path
from importlib import import_module
from typing import Dict, Any

from metrics import MetricsContainer


def _get_model_class(name: str):
    """Dynamically import a model class from the package."""
    modules = [
        "src.models.classical",
        "src.models.hybrid",
        "src.models.quantum",
    ]
    for mod in modules:
        module = import_module(mod)
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"Unknown model class '{name}'")


def _best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the checkpoint with the lowest validation loss."""
    if not checkpoint_dir.exists():
        return None
    best_path = None
    best_loss = float("inf")
    pattern = re.compile(r"loss=([0-9.]+)")
    for ckpt in checkpoint_dir.glob("*.ckpt"):
        match = pattern.search(ckpt.name)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_path = ckpt
    return best_path


def load_model(path: str, *, best: bool = True) -> Dict[str, Any]:
    """Load all seeded models found in ``path``.

    Parameters
    ----------
    path:
        Directory corresponding to ``model_config_name`` containing seed folders.
    best:
        When ``True`` loads the best checkpoint found in the ``checkpoints``
        folder. Otherwise loads ``final.ckpt``.

    Returns
    -------
    Dict[str, LightningModule]
        Mapping of seed value to the loaded model instance.
    """
    path = Path(path)
    model_class = path.parent.name
    model_cls = _get_model_class(model_class)

    models = {}
    for seed_dir in path.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue
        ckpt = seed_dir / "final.ckpt"
        if best:
            candidate = _best_checkpoint(seed_dir / "checkpoints")
            if candidate is not None:
                ckpt = candidate
        model = model_cls.load_from_checkpoint(str(ckpt), map_location="cpu")
        models[seed_dir.name] = model
    return models


def load_metrics(path: str, *, best: bool = True) -> Dict[str, MetricsContainer]:
    """Load metrics for each seed found in ``path``.

    The ``best`` argument is accepted for API compatibility but currently has no
    effect since metrics are stored independently of checkpoints.
    """
    path = Path(path)
    metrics = {}
    for seed_dir in path.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue
        mc = MetricsContainer(str(seed_dir))
        mc.load_metrics(str(seed_dir))
        metrics[seed_dir.name] = mc
    return metrics


def load_experiment(path: str, *, best: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load an entire experiment structure.

    Parameters
    ----------
    path:
        Root experiment directory that contains model class folders.
    best:
        When ``True`` loads the best checkpoints for each model; otherwise the
        final checkpoints.
    """
    path = Path(path)
    experiment: Dict[str, Dict[str, Any]] = {}
    for model_class_dir in path.iterdir():
        if not model_class_dir.is_dir():
            continue
        for config_dir in model_class_dir.iterdir():
            if not config_dir.is_dir():
                continue
            experiment[config_dir.name] = load_model(str(config_dir), best=best)
    return experiment
