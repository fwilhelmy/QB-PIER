"""Class distribution visualisations."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only

__all__ = ["plot_class_distribution"]


@rank_zero_only
def plot_class_distribution(dataset, logdir: str = "logs/visualisation"):
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dataset.classes)), class_counts, color="blue", alpha=0.7)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Dataset")
    plt.xticks(range(len(dataset.classes)), dataset.classes)

    os.makedirs(logdir, exist_ok=True)
    plt.savefig(os.path.join(logdir, "class_distribution.png"))
    plt.close()
