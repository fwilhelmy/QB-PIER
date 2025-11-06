"""Confusion matrix visualisations."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lightning.pytorch.utilities import rank_zero_only
from sklearn.metrics import confusion_matrix

__all__ = ["plot_confusion_matrix"]


@rank_zero_only
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    root_dir: str,
    source_model: str,
    normalize: bool = True,
):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plt.title(f"{source_model} @ max ε: {title}")
    plt.tight_layout()

    out_dir = os.path.join(root_dir, source_model)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
