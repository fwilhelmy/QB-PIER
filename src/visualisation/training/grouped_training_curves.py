"""Visualisations comparing multiple trained models."""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.ticker import MaxNLocator

from ..utils import line_with_std

__all__ = ["plot_models_together"]


@rank_zero_only
def plot_models_together(trained_models, fig_dir: str = "logs"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex="col", sharey="row")
    ax_train_loss, ax_val_loss = ax[0]
    ax_train_acc, ax_val_acc = ax[1]

    ax_train_loss.set_title("Training", fontweight="bold")
    ax_val_loss.set_title("Validation", fontweight="bold")
    ax_train_loss.set_yscale("log")
    ax_val_loss.set_yscale("log")

    ax_train_loss.set_ylabel("Loss")
    ax_train_acc.set_ylabel("Accuracy (%)")
    ax_val_loss.set_ylabel("")
    ax_val_acc.set_ylabel("")

    for name, seed_dict in trained_models.items():
        line_with_std(
            ax_train_loss,
            seed_dict=seed_dict,
            metric_name="train/loss",
            step_name="train/step",
            label=name,
        )
        line_with_std(
            ax_val_loss,
            seed_dict=seed_dict,
            metric_name="val/loss",
            step_name="val/step",
            label=name,
        )
        line_with_std(
            ax_train_acc,
            seed_dict=seed_dict,
            metric_name="train/acc",
            step_name="train/step",
            label=name,
        )
        line_with_std(
            ax_val_acc,
            seed_dict=seed_dict,
            metric_name="val/acc",
            step_name="val/step",
            label=name,
        )

    first_model = list(next(iter(trained_models.values())).values())[0]
    epoch_steps = first_model.metrics["epoch"]
    n_epochs = len(epoch_steps)
    epoch_nums = list(range(1, n_epochs + 1))
    max_ticks = 8
    if n_epochs > max_ticks:
        stride = math.ceil(n_epochs / max_ticks)
        idxs = list(range(0, n_epochs, stride))
        if idxs[-1] != n_epochs - 1:
            idxs.append(n_epochs - 1)
        bottom_steps = [epoch_steps[i] for i in idxs]
        bottom_labels = [epoch_nums[i] for i in idxs]
    else:
        bottom_steps = epoch_steps
        bottom_labels = epoch_nums

    for subax, (row, col) in zip(
        (ax_train_loss, ax_val_loss, ax_train_acc, ax_val_acc),
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ):
        subax.set_xticks(bottom_steps)
        subax.set_xticklabels(bottom_labels)
        subax.set_xlabel("Epoch" if row == 1 else "")

        ax_top = subax.twiny()
        ax_top.set_xlim(subax.get_xlim())
        ax_top.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        if row == 1:
            ax_top.set_xlabel("")
            ax_top.set_xticklabels([])
        else:
            ax_top.set_xlabel("")
        ax_top.xaxis.set_label_position("top")
        ax_top.xaxis.tick_top()

        if col == 1:
            subax.set_ylabel("")

        subax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "models_together.png"))
    plt.close(fig)
