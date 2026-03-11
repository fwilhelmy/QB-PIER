"""Visualisations comparing multiple trained models."""

from __future__ import annotations

import math
import os
import logging

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.ticker import MaxNLocator

from visualisation.utils import line_with_std, metric_stats_from_seeds, format_x_axis

import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

log = logging.getLogger(__name__)

@rank_zero_only
def plot_models_together(trained_models, fig_dir: str = None, file_name: str = "models_together.png", show: bool = False):
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

    metrics = (
        ("train", {"loss": ax_train_loss, "acc": ax_train_acc}),
        ("val", {"loss": ax_val_loss, "acc": ax_val_acc}),
    )
    for name, seed_dict in trained_models.items():
        try:
            for group, axes in metrics:
                for metric, ax in axes.items():
                    x, mean, std = metric_stats_from_seeds(seed_dict, metric=metric, group=group)
                    line_with_std(ax, x=x, mean=mean, std=std, label=name)
        except ValueError as e:
            log.error(f"Skipping model '{name}' due to error: {e}")
            continue
                
    first_model = list(next(iter(trained_models.values())).values())[0]
    epoch_steps = np.array(first_model.metrics["epoch"], dtype=float)

    for subax, (row, col) in zip(
        (ax_train_loss, ax_val_loss, ax_train_acc, ax_val_acc),
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ):
        ax_top = format_x_axis(subax, epoch_steps, bottom="epoch", top="step", nbins=8)
        subax.set_xlabel("Epoch" if row == 1 else "")

        if ax_top is not None:
            ax_top.set_xlabel("")
            if row == 1:
                ax_top.set_xticklabels([])

        if col == 1:
            subax.set_ylabel("")

        subax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if show:
        plt.show()
    if fig_dir is not None or fig_dir != "":
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, file_name))
    plt.close(fig)

