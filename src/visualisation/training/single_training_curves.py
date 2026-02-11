"""Training curve visualisations."""

from __future__ import annotations

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.ticker import FuncFormatter, MaxNLocator

from visualisation.utils import line_with_std, metric_stats_from_seeds

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

    epoch_steps = np.array(first_model.metrics["epoch"])
    epoch_numbers = np.arange(1, len(epoch_steps) + 1)

    for idx, ax in enumerate((ax_loss, ax_acc, ax_w2)):
        ax_epoch = ax.twiny()
        ax_epoch.set_xlim(ax.get_xlim())
        ax_epoch.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

        def epoch_fmt(x, pos):
            idx = np.searchsorted(epoch_steps, x, side="right") - 1
            if 0 <= idx < len(epoch_numbers):
                return str(epoch_numbers[idx])
            return ""

        ax_epoch.xaxis.set_major_formatter(FuncFormatter(epoch_fmt))
        ax_epoch.xaxis.tick_top()

        ticks = ax_epoch.get_xticks()
        ticks = ticks[(ticks >= epoch_steps[0]) & (ticks <= epoch_steps[-1])]
        ticks = np.unique(np.concatenate([ticks, [epoch_steps[0], epoch_steps[-1]]]))
        ax_epoch.set_xticks(ticks)

        if idx == 0:
            ax_epoch.set_xlabel("Epoch")
            ax_epoch.xaxis.set_label_position("top")
        else:
            ax_epoch.set_xticklabels([])

        ax_epoch.grid(axis="x", linestyle="-", linewidth=0.8)

    fig.suptitle(f"Training Curves — {model_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir = fig_dir or "logs"
    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, f"training_{model_name}.png")
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path
