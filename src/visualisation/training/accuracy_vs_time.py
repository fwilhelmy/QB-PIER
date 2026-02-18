"""Plot architecture accuracy versus wall-clock time from loaded seed dictionaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

from visualisation.utils import line_with_std, metric_stats_from_seeds


X_UNITS_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}

def generate_arch_accuracy_vs_time(
    models: Mapping[str, Mapping[int, object]],
    out_dir: str | Path,
    include_train: bool = False,
    x_units: str = "minutes",
) -> tuple[Path, Path]:
    if x_units not in X_UNITS_TO_SECONDS:
        raise ValueError(f"Unsupported x_units '{x_units}'. Expected one of: {sorted(X_UNITS_TO_SECONDS)}")
    if not models:
        raise ValueError("No models were provided.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_seconds = X_UNITS_TO_SECONDS[x_units]
    fig, ax = plt.subplots(figsize=(10, 6))
    resolved_models: list[dict[str, object]] = []
    plotted_any = False

    for model_name, seed_dict in models.items():
        if not seed_dict:
            continue

        try:
            x, val_mean, val_std = metric_stats_from_seeds(seed_dict, metric="acc", group="val", x_axis="time")
            line_with_std(
                ax,
                x=x / unit_seconds,
                mean=val_mean,
                std=val_std,
                label=f"{model_name} val/acc",
                linewidth=2,
            )
            plotted_any = True
        except Exception as e:
            print(f"Error processing val/acc for {model_name}: {e}")
            continue

        model_meta: dict[str, object] = {
            "model_name": model_name,
            "seeds": sorted(seed_dict.keys()),
            "points": int(val_mean.shape[0]),
            "metrics": ["val/acc"],
        }

        if include_train:
            try:
                train_x, train_mean, train_std = metric_stats_from_seeds(
                    seed_dict,
                    metric="acc",
                    group="train",
                    x_axis="time",
                )
                line_with_std(
                    ax,
                    x=train_x / unit_seconds,
                    mean=train_mean,
                    std=train_std,
                    label=f"{model_name} train/acc",
                    linestyle="--",
                    alpha=0.8,
                )
                model_meta["metrics"].append("train/acc")
            except Exception as e:
                print(f"Error processing train/acc for {model_name}: {e}")

        resolved_models.append(model_meta)

    if not plotted_any:
        raise RuntimeError("No model could be plotted. Check loaded model metrics/tensorboard logs.")

    ax.set_xlabel(f"Elapsed time ({x_units})")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs wall-clock time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    figure_path = out_dir / "arch_accuracy_vs_time.png"
    fig.savefig(figure_path)
    plt.close(fig)

    sidecar_path = out_dir / "arch_accuracy_vs_time.json"
    payload = {
        "include_train": include_train,
        "x_units": x_units,
        "models": list(models.keys()),
        "resolved_models": resolved_models,
        "figure_path": str(figure_path),
        "metrics_source": "metric_stats_from_seeds",
    }
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return figure_path, sidecar_path
