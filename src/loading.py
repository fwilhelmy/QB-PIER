from __future__ import annotations

"""Utility helpers for loading models and experiment data."""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from metrics import MetricsContainer
from models.ModelFactory import ModelFactory

log = logging.getLogger(__name__)


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

def _best_checkpoint(ckpt_dir: str) -> str:
    """Return path to the best checkpoint inside ``ckpt_dir``."""
    if not os.path.exists(ckpt_dir):
        return None
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
    Dict[str, Any]
        Mapping of seed value to the loaded model instance.
    """

    with open(os.path.join(path, ".model"), "r") as f:
        model_name = f.read().strip()
    models = {}
    for seed in os.listdir(path):
        seed_dir = os.path.join(path, seed)
        if not os.path.isdir(seed_dir) or os.path.basename(seed_dir) == ".hydra" or not seed.isdigit():
            continue
        ckpt = os.path.join(seed_dir, "final.ckpt")
        if best:
            candidate = _best_checkpoint(os.path.join(seed_dir, "checkpoints"))
            if candidate is not None:
                ckpt = candidate
        model = ModelFactory.load(model_name, checkpoint=str(ckpt), map_location=map_location, log_dir=seed_dir)
        models[int(seed)] = model
    return models


def load_metrics(path: str) -> Dict[Any, MetricsContainer]:
    """Load metrics for all seeds of a model configuration."""
    metrics = {}
    for seed in sorted(os.listdir(path)):
        seed_dir = os.path.join(path, seed)
        if not os.path.isdir(seed_dir) or not seed.isdigit():
            continue
        container = MetricsContainer(seed_dir)
        container.load_metrics(seed_dir)
        metrics[int(seed)] = container
    return metrics

def load_experiment(path: str, *, load: str = "final", map_location: str = "cpu") -> Dict[str, Dict[Any, Any]]:
    experiment = {}

    for model_path in tqdm(crawl_experiments(path), desc="Loading experiment"):
        name = os.path.relpath(model_path, path)

        if load == "metrics":
            experiment[name] = load_metrics(model_path)
        elif load == "final":
            experiment[name] = load_model(model_path, best=False, map_location=map_location)
        elif load == "best":
            experiment[name] = load_model(model_path, best=True, map_location=map_location)
        else:
            raise ValueError(f"Unknown load type '{load}'")

    return experiment


def load_attacks(path: str) -> pd.DataFrame:
    """Load attack results from ``path``.

    Parameters
    ----------
    path:
        CSV file containing previously computed attack results.

    Returns
    -------
    pandas.DataFrame
        Loaded attack results.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the CSV file is missing required columns.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Results path '{path}' does not exist.")

    results = pd.read_csv(path)
    expected_cols = {
        "seed",
        "source_model",
        "target_model",
        "epsilon",
        "accuracy",
    }
    missing_cols = expected_cols - set(results.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"Results file at '{path}' is missing required columns: {missing}"
        )

    return results

from analytics.training import generate_gradient_flow_plots, plot_models_together
if __name__ == "__main__":
    log_root = "logs\ByModel\Classical"
    os.makedirs(log_root, exist_ok=True)
    data = load_experiment(log_root, load="final")
    log.info("Loaded data for %s model configurations.", len(data))
    # for model_name, model_data in data.items():
    #     log.info("Generating gradient flow plots for %s.", model_name)
    #     generate_gradient_flow_plots(model_data, fig_dir=os.path.join(log_root, model_name))
    plot_models_together(data, fig_dir=log_root)
