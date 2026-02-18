"""Plot per-model generalization gap curves from loaded model dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


_REQUIRED_METRICS = {
    "loss": ("train_loss", "val_loss"),
    "acc": ("train_acc", "val_acc"),
}


def _mean_curve_for_metric(seed_dict: Mapping[int, object], metric_name: str) -> np.ndarray | None:
    curves: list[np.ndarray] = []

    for _, model in seed_dict.items():
        values = model.metrics[metric_name]
        if values is None:
            continue

        curve = np.asarray(values, dtype=np.float64)
        if curve.size == 0:
            continue
        curves.append(curve)

    if not curves:
        return None

    min_len = min(curve.size for curve in curves)
    aligned = np.vstack([curve[:min_len] for curve in curves])
    return aligned.mean(axis=0)


def _build_gap_curve(seed_dict: Mapping[int, object], gap_type: str) -> np.ndarray | None:
    train_metric, val_metric = _REQUIRED_METRICS[gap_type]
    train_curve = _mean_curve_for_metric(seed_dict, train_metric)
    val_curve = _mean_curve_for_metric(seed_dict, val_metric)

    if train_curve is None or val_curve is None:
        return None

    min_len = min(train_curve.size, val_curve.size)
    if min_len == 0:
        return None

    train_curve = train_curve[:min_len]
    val_curve = val_curve[:min_len]

    if gap_type == "loss":
        return val_curve - train_curve
    return train_curve - val_curve


def generate_arch_generalization_gap(
    models: Mapping[str, Mapping[int, object]],
    out_dir: str | Path,
    gap_type: str = "loss",
    output_format: str = "pdf",
    metrics_source: str = "model_dict",
    tb_size_guidance: int = 0,
) -> tuple[Path, Path]:
    if gap_type not in _REQUIRED_METRICS:
        raise ValueError(f"Unsupported gap_type '{gap_type}'. Expected one of: {tuple(_REQUIRED_METRICS)}")

    if not models:
        raise ValueError("No models were provided.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    resolved_models: list[dict[str, object]] = []

    for model_name, seed_dict in models.items():
        if not seed_dict:
            continue

        gap_curve = _build_gap_curve(seed_dict=seed_dict, gap_type=gap_type)
        if gap_curve is None:
            continue

        epochs = np.arange(1, gap_curve.size + 1)
        plt.plot(epochs, gap_curve, label=model_name, linewidth=1.8)

        resolved_models.append(
            {
                "label": model_name,
                "seeds": sorted(seed_dict.keys()),
                "points": int(gap_curve.size),
                "resolved_metrics": {
                    "train": _REQUIRED_METRICS[gap_type][0],
                    "val": _REQUIRED_METRICS[gap_type][1],
                },
            }
        )

    if not resolved_models:
        raise RuntimeError("No model could be plotted. Check loaded model metrics.")

    ylabel = "val loss - train loss" if gap_type == "loss" else "train acc - val acc"
    plt.title(f"Generalization gap across epochs ({gap_type})")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    fig_path = out_dir / f"arch_generalization_gap.{output_format}"
    sidecar_path = out_dir / "arch_generalization_gap.json"

    plt.savefig(fig_path)
    plt.close()

    sidecar = {
        "metrics_source": metrics_source,
        "gap_type": gap_type,
        "tb_size_guidance": tb_size_guidance,
        "models": list(models.keys()),
        "resolved_models": resolved_models,
        "output_figure": str(fig_path),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return fig_path, sidecar_path
