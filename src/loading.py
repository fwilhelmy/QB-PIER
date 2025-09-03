from __future__ import annotations

import os
import re
from importlib import import_module
from typing import Dict, Any, List

from lightning import LightningModule

from metrics import MetricsContainer
from tqdm import tqdm


def crawl_experiments(path: str) -> List[str]:
    """Walk ``path`` and return directories containing a ``.model`` file.

    Parameters
    ----------
    path:
        Root directory to search for experiments.

    Returns
    -------
    List[str]
        Sorted list of absolute paths to experiment directories.
    """
    experiments: List[str] = []
    for root, _, files in os.walk(path):
        if ".model" in files:
            experiments.append(root)
    experiments.sort()
    return experiments

def _resolve_class(meta: str) -> LightningModule:
    """Dynamically import a model class from the package."""
    model_mod, model_cls = meta.rsplit(".",1)
    module = import_module(model_mod)
    if hasattr(module, model_cls):
        return getattr(module, model_cls)
    raise ValueError(f"Unknown model class '{meta}'")


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
    with open(os.path.join(path, ".model"), "r") as f:
        meta = f.read().strip()
    model_cls = _resolve_class(meta)
    models = {}
    for seed in os.listdir(path):
        seed_dir = os.path.join(path, seed)
        if not os.path.isdir(seed_dir) or os.path.basename(seed_dir) == ".hydra":
            continue
        ckpt = os.path.join(seed_dir, "final.ckpt")
        if best:
            candidate = _best_checkpoint(os.path.join(seed_dir, "checkpoints"))
            if candidate is not None:
                ckpt = candidate
        model = model_cls.load_from_checkpoint(str(ckpt), map_location, log_dir=seed_dir)
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

def load_experiment(path: str, *, load: str = "final", map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
    """Load every experiment under ``path`` discovered via ``crawl_experiments``.

    The returned dictionary is keyed by the relative directory of each model
    with respect to ``path``.
    """
    experiment = {}
    for root in tqdm(crawl_experiments(path), desc="Loading experiment"):
        name = os.path.relpath(root, path)

        if load == "metrics":
            experiment[name] = load_metrics(root)
        elif load == "final":
            experiment[name] = load_model(root, best=False, map_location=map_location)
        elif load == "best":
            experiment[name] = load_model(root, best=True, map_location=map_location)
        else:
            raise ValueError(f"Unknown load type '{load}'")
    return experiment

