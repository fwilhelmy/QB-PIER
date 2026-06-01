from __future__ import annotations

from pathlib import Path
import logging
from typing import Mapping

import matplotlib.pyplot as plt

from analytics.utils import line_with_std, metric_stats_from_seeds
from analytics.metrics_io import ModelMetrics

log = logging.getLogger(__name__)

_UNITS_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}


def accuracy_vs_time(
    models: ModelMetrics,
    out_dir: str | Path = "logs",
    *,
    include_train: bool = False,
    x_units: str = "minutes",
) -> Path:
    """
    Plot val/acc (and optionally train/acc) versus wall-clock time.

    - Uses TensorBoard wall_time via `metric_stats_from_seeds(..., x_axis="time")`.
    - Saves one PNG and returns its path.
    """
    if not models:
        raise ValueError("No models provided.")

    if x_units not in _UNITS_TO_SECONDS:
        raise ValueError(f"x_units must be one of {sorted(_UNITS_TO_SECONDS)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_seconds = _UNITS_TO_SECONDS[x_units]
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_any = False

    for model_name, seed_dict in models.items():
        if not seed_dict:
            continue

        # --- Validation accuracy ---
        try:
            x, mean, std = metric_stats_from_seeds(seed_dict, metric="acc", group="val", x_axis="time")
            line_with_std(ax, x=x / unit_seconds, mean=mean, std=std, label=f"{model_name} val/acc", linewidth=2)
            plotted_any = True
        except Exception as e:
            log.error(f"Failed to plot val/acc for model '{model_name}': {e}")
            continue

        # --- Optional training accuracy ---
        if include_train:
            try:
                x, mean, std = metric_stats_from_seeds(seed_dict, metric="acc", group="train", x_axis="time")
                line_with_std(ax, x=x / unit_seconds, mean=mean, std=std, label=f"{model_name} train/acc", linestyle="--", alpha=0.8)
                plotted_any = True
            except Exception as e:
                log.warning(f"Failed to plot train/acc for model '{model_name}': {e}")
                pass

    if not plotted_any:
        raise RuntimeError("Nothing was plotted (missing metrics or tensorboard logs?).")

    ax.set_xlabel(f"Elapsed time ({x_units})")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs wall-clock time")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path = out_dir / "accuracy_vs_time.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
