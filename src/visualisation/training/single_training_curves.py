"""Training curve visualisations."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.ticker import FuncFormatter, MaxNLocator

from visualisation.utils import line_with_std

@rank_zero_only
def plot_training_curves(model_name, seed_dict, fig_dir=None):
    """Plot averaged loss, accuracy and weights norm over multiple seeds."""

    seeds = sorted(seed_dict.keys())
    if not seeds:
        raise ValueError("No seeds were provided")
    first_model = seed_dict[seeds[0]]

    fig, (ax_loss, ax_acc, ax_w2) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    line_with_std(ax_loss, seed_dict=seed_dict, metric_name="train_loss", label="Train Loss")
    line_with_std(ax_loss, seed_dict=seed_dict, metric_name="val_loss", label="Val Loss")
    ax_loss.set_ylabel("Loss (log)")
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(axis="x", linestyle="--", linewidth=0.5)

    line_with_std(ax_acc, seed_dict=seed_dict, metric_name="train_acc", label="Train Acc")
    line_with_std(ax_acc, seed_dict=seed_dict, metric_name="val_acc", label="Val Acc")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.legend()
    ax_acc.grid(axis="x", linestyle="--", linewidth=0.5)

    line_with_std(ax_w2, seed_dict=seed_dict, metric_name="train_weights_l2", label="Weights L2")
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
