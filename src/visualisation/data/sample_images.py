"""Dataset sample visualisations."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only

from LoggingService import LoggingService

__all__ = ["plot_sample_images"]


@rank_zero_only
def plot_sample_images(dataset, x: int = 10, rows: int = 2, cols: int = 5, logdir: str = "logs/visualisation"):
    indices = random.sample(range(len(dataset)), x)
    images, labels = zip(*[dataset[i] for i in indices])

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i // cols, i % cols]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.tight_layout()

    LoggingService().save_figure(fig, "sample_images", logdir)
    plt.close(fig)
