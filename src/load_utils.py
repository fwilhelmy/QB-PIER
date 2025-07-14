from __future__ import annotations

import os
import re
from importlib import import_module
from typing import Dict, Any
import json

from lightning import LightningModule

from metrics import MetricsContainer


def _resolve_class(name: str) -> LightningModule:
    """Dynamically import a model class from the package."""
    name = name.rsplit("\'")[1]
    model_mod, model_cls, _ = name.rsplit(".",2)
    module = import_module(model_mod)
    if hasattr(module, model_cls):
        return getattr(module, model_cls)
    raise ValueError(f"Unknown model class '{name}'")


def _best_checkpoint(ckpt_dir: str) -> str:
    """Return path to the best checkpoint inside ``ckpt_dir``."""
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError("No checkpoint files found")

    def score(name: str) -> float:
        m = re.search(r"loss=(\d+.\d+)", name)
        return float(m.group(1)) if m else float("inf")

    ckpts.sort(key=score)
    return os.path.join(ckpt_dir, ckpts[0]) 

def load_model(path: str, *, best: bool = True, map_location: str = "cpu") -> Dict[str, Any]:
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
    summary_path = os.path.join(path, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    model_cls = _resolve_class(summary["model"])
    models = {}
    for seed in os.listdir(path):
        seed_dir = os.path.join(path, seed)
        if not os.path.isdir(seed_dir): continue
        ckpt = os.path.join(seed_dir, "final.ckpt")
        if best:
            candidate = _best_checkpoint(os.path.join(seed_dir, "checkpoints"))
            if candidate is not None:
                ckpt = candidate
        model = model_cls.load_from_checkpoint(str(ckpt), map_location)
        models[int(seed)] = model
    return models


def load_metrics(path: str) -> Dict[Any, MetricsContainer]:
    """Load metrics for all seeds of a model configuration."""
    metrics = {}
    for seed in sorted(os.listdir(path)):
        seed_dir = os.path.join(path, seed)
        if not os.path.isdir(seed_dir): continue
        container = MetricsContainer(seed_dir)
        container.load_metrics(seed_dir)
        metrics[int(seed)] = container
    return metrics

def load_experiment(path: str, best: bool = True) -> Dict[str, Dict[Any, Any]]:
    """Load every model in an experiment directory."""
    experiment = {}
    for model_cls in os.listdir(path):
        cls_dir = os.path.join(path, model_cls)
        if not os.path.isdir(cls_dir): continue
        for config in os.listdir(cls_dir):
            config_dir = os.path.join(cls_dir, config)
            if not os.path.isdir(config_dir): continue
            experiment[config] = load_model(config_dir, best=best)
    return experiment
