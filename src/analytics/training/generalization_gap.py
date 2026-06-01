from __future__ import annotations

from pathlib import Path
from typing import Mapping
import logging

import matplotlib.pyplot as plt
import numpy as np

from analytics.utils import line_with_std, metric_stats_from_seeds
from analytics.metrics_io import ModelMetrics

log = logging.getLogger(__name__)

def generalization_gap(
    models: ModelMetrics,
    out_dir: str | Path = "logs",
    *,
    gap_type: str = "loss",          # "loss" or "acc"
) -> Path:
    """
    Plot per-model generalization gap curves as a function of STEP (real x).

    Gap definition:
      - loss:  val/loss - train/loss
      - acc:   train/acc - val/acc

    Saves one PNG and returns its path.

    Notes:
      - Uses metric_stats_from_seeds for mean/std per metric.
      - Gap std is approximated as sqrt(std_train^2 + std_val^2). This is simple and
        usually fine; if you want exact (per-seed gap then stats), say so.
    """
    if gap_type not in {"loss", "acc"}:
        raise ValueError("gap_type must be 'loss' or 'acc'")
    if not models:
        raise ValueError("No models provided.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False

    for model_name, seed_dict in models.items():
        if not seed_dict:
            continue

        try:
            # Pull both curves in the same x-domain (step by default).
            x_tr, mean_tr, std_tr = metric_stats_from_seeds(seed_dict, metric=gap_type, group="train", x_axis="step")
            x_va, mean_va, std_va = metric_stats_from_seeds(seed_dict, metric=gap_type, group="val",   x_axis="step")

            # Align on common prefix (robust if one series is slightly shorter).
            n = int(min(len(x_tr), len(x_va), len(mean_tr), len(mean_va), len(std_tr), len(std_va)))
            if n <= 0:
                continue

            x = np.asarray(x_tr[:n], dtype=float)  # step-space x
            mean_tr = np.asarray(mean_tr[:n], dtype=float)
            mean_va = np.asarray(mean_va[:n], dtype=float)
            std_tr = np.asarray(std_tr[:n], dtype=float)
            std_va = np.asarray(std_va[:n], dtype=float)

            if gap_type == "loss":
                gap_mean = mean_va - mean_tr
            else:  # acc
                gap_mean = mean_tr - mean_va

            gap_std = np.sqrt(std_tr**2 + std_va**2)

            line_with_std(ax, x=x, mean=gap_mean, std=gap_std, label=model_name, linewidth=2)
            plotted_any = True

        except Exception as e:
            log.error(f"Failed to plot generalization gap for model '{model_name}': {e}")
            continue

    if not plotted_any:
        raise RuntimeError("Nothing was plotted (missing train/val metrics across all models).")

    ax.set_xlabel("Step")
    ylabel = "val loss - train loss" if gap_type == "loss" else "train acc - val acc"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Generalization gap vs step ({gap_type})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path = out_dir / f"generalization_gap_{gap_type}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
