"""Training curve visualisations."""

from __future__ import annotations

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities import rank_zero_only

from visualisation.utils import line_with_std, metric_stats_from_seeds, format_x_axis

log = logging.getLogger(__name__)

@rank_zero_only
def plot_training_curves(model_name, seed_dict, fig_dir=None):
    """Plot averaged loss, accuracy and weights norm over multiple seeds."""

    valid_seeds = []
    for seed, model in seed_dict.items():
        required_metrics = {"train/loss", "val/loss", "train/acc", "val/acc", "train/weights_l2"}
        if all(model.metrics[metric] is not None for metric in required_metrics):
            valid_seeds.append(seed)
        else:
            missing = required_metrics - set(model.metrics._raw)
            log.warning(f"Seed {seed} is missing required metrics: {missing}")
    if not valid_seeds:
        raise ValueError("No seeds with required metrics found for plotting")
    first_model = seed_dict[valid_seeds[0]]

    fig, (ax_loss, ax_acc, ax_w2) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    train_metrics = ("train", {"loss": ax_loss, "acc": ax_acc, "weights_l2": ax_w2})
    val_metrics = ("val", {"loss": ax_loss, "acc": ax_acc})
    for group, metrics in (train_metrics, val_metrics):
        for metric, ax in metrics.items():
            x, mean, std = metric_stats_from_seeds(seed_dict, metric=metric, group=group)
            line_with_std(ax, x=x, mean=mean, std=std, label=f"{group.title()} {metric.title()}")

    ax_loss.set_ylabel("Loss (log)")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(axis="x", linestyle="--", linewidth=0.5)

    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.legend()
    ax_acc.grid(axis="x", linestyle="--", linewidth=0.5)

    ax_w2.set_ylabel("Weights L2 Norm")
    ax_w2.set_xlabel("Step")
    ax_w2.legend()
    ax_w2.grid(axis="x", linestyle="--", linewidth=0.5)

    epoch_steps = np.array(first_model.metrics["epoch"], dtype=float)

    for idx, _ax in enumerate((ax_loss, ax_acc, ax_w2)):
        ax_top = format_x_axis(_ax, epoch_steps, bottom="step", top="epoch", nbins=10)
        if idx == 0:
            ax_top.set_xlabel("Epoch")
            ax_top.xaxis.set_label_position("top")
        else:
            ax_top.set_xticklabels([])
            ax_top.set_xlabel("")
        ax_top.grid(axis="x", linestyle="-", linewidth=0.8)

    fig.suptitle(f"Training Curves — {model_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir = fig_dir or "logs"
    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, f"training_{model_name}.png")
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path
