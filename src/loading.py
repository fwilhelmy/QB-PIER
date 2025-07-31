from __future__ import annotations

import os
import re
from importlib import import_module
from typing import Dict, Any

from lightning import LightningModule

from metrics import MetricsContainer, LoggingService
from tqdm import tqdm

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
    """Recursively load every model under ``path`` that contains a ``.model``
    anchor file.

    The returned dictionary is keyed by the relative directory of each model
    with respect to ``path``.
    """
    experiment = {}
    for root, _, files in tqdm(list(os.walk(path)), desc="Loading experiment"):
        if ".model" in files:
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


def load_from_dict(models: Dict[str, str], *, load: str = "final",
                   map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
    """Load multiple models from a mapping of names to paths using different load types.

    Parameters
    ----------
    models:
        Mapping of the desired name to the directory containing the model.
    load:
        Load type: "metrics" to load metrics, "final" to load final checkpoints,
        or "best" to load the best checkpoint.
    map_location:
        Device mapping for Lightning when loading the checkpoints.

    Returns
    -------
    Dict[str, Dict[Any, LightningModule] or MetricsContainer]
        Dictionary keyed by the provided names with the loaded models/metrics per seed.
    """
    loaded = {}
    for name, path in tqdm(models.items(), desc="Loading models"):
        if load == "metrics":
            loaded[name] = load_metrics(path)
        elif load == "final":
            loaded[name] = load_model(path, best=False, map_location=map_location)
        elif load == "best":
            loaded[name] = load_model(path, best=True, map_location=map_location)
        else:
            raise ValueError(f"Unknown load type '{load}'")
    return loaded

from main import run_attacks
from visualisation.training import plot_training_curves, plot_models_together
from visualisation.attacks import plot_transfer_attack_effects
import sys
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python loading.py <root_dir>")
        sys.exit(1)
    root_dir = sys.argv[1]
    ls = LoggingService(root_dir)
    models = load_experiment(root_dir, load="final", map_location="cpu")
    # for path, seeds in models.items():
    #     plot_training_curves("Model", seeds, path)
    # plot_models_together(models)
    attack_results = run_attacks(models, "mnist", "cpu")
    os.makedirs(os.path.join(ls.root_dir, "attacks"), exist_ok=True)
    attack_path = os.path.join(ls.root_dir, "attacks")
    attack_results.to_csv(os.path.join(attack_path, "results.csv"), index=False)
    attack_results.to_json(os.path.join(attack_path, "results.json"), index=False)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")
