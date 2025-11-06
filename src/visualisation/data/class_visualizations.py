"""Per-class sample, mean and deviation visualisations."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities import rank_zero_only

from LoggingService import LoggingService

__all__ = ["plot_class_visualizations"]


@rank_zero_only
def plot_class_visualizations(dataset, logdir: str = "logs/visualisation"):
    mean_images = []
    std_images = []
    sample_images = []

    for i in range(len(dataset.classes)):
        class_images = torch.stack([img for img, label in dataset if label == i])
        mean_images.append(class_images.mean(dim=0))
        std_images.append(class_images.std(dim=0))
        sample_images.append(class_images[random.randint(0, len(class_images) - 1)])

    fig, axes = plt.subplots(4, len(dataset.classes), figsize=(15, 8))
    for i in range(len(dataset.classes)):
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Class {dataset.classes[i]}")

        axes[1, i].imshow(sample_images[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Sample")

        axes[2, i].imshow(mean_images[i].squeeze(), cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title("Mean")

        axes[3, i].imshow(std_images[i].squeeze(), cmap="gray")
        axes[3, i].axis("off")
        axes[3, i].set_title("STD")

    plt.tight_layout()
    LoggingService().save_figure(fig, "class_visualizations", logdir)
    plt.close(fig)
