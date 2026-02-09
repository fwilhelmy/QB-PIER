"""Visualisations comparing multiple trained models."""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.ticker import MaxNLocator

from .utils import line_with_std

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

    def _ensure_array(data, dtype=None):
        if isinstance(data, dict) or data is None:
            return np.array([], dtype=dtype if dtype is not None else float)
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    for name, seed_dict in trained_models.items():
        losses_tr, losses_val, accs_tr, accs_val = [], [], [], []
        train_steps_ref, val_steps_ref = None, None

        for s in seed_dict.values():
            metrics = s.metrics

            train_steps = _ensure_array(metrics["train/step"])
            val_steps = _ensure_array(metrics["val/step"])
            train_loss = _ensure_array(metrics["train/loss"])
            val_loss = _ensure_array(metrics["val/loss"])
            train_acc = _ensure_array(metrics["train/acc"])
            val_acc = _ensure_array(metrics["val/acc"])

            if train_loss.shape != train_steps.shape or train_acc.shape != train_steps.shape:
                raise ValueError("Training metrics and steps lengths do not match")
            if val_loss.shape != val_steps.shape or val_acc.shape != val_steps.shape:
                raise ValueError("Validation metrics and steps lengths do not match")

            if train_steps_ref is None:
                train_steps_ref = train_steps
            elif train_steps.shape != train_steps_ref.shape or not np.array_equal(train_steps, train_steps_ref):
                raise ValueError("Inconsistent training step counts across seeds")

            if val_steps_ref is None:
                val_steps_ref = val_steps
            elif val_steps.shape != val_steps_ref.shape or not np.array_equal(val_steps, val_steps_ref):
                raise ValueError("Inconsistent validation step counts across seeds")

            losses_tr.append(train_loss)
            losses_val.append(val_loss)
            accs_tr.append(train_acc)
            accs_val.append(val_acc)

        if (
            train_steps_ref is None
            or val_steps_ref is None
            or train_steps_ref.size == 0
            or val_steps_ref.size == 0
        ):
            continue

        Ltr = np.vstack(losses_tr)
        Lval = np.vstack(losses_val)
        Atr = np.vstack(accs_tr)
        AVal = np.vstack(accs_val)

        line_with_std(ax_train_loss, train_steps_ref, Ltr.mean(0), Ltr.std(0), label=name)
        line_with_std(ax_val_loss, val_steps_ref, Lval.mean(0), Lval.std(0), label=name)
        line_with_std(ax_train_acc, train_steps_ref, Atr.mean(0), Atr.std(0), label=name)
        line_with_std(ax_val_acc, val_steps_ref, AVal.mean(0), AVal.std(0), label=name)

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
