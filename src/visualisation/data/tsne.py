"""Dataset t-SNE visualisations."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities import rank_zero_only
from sklearn.manifold import TSNE

from LoggingService import LoggingService

__all__ = ["plot_tsne"]


@rank_zero_only
def plot_tsne(dataset, subset_size: int = 1000, logdir: str = "logs/visualisation"):
    indices = random.sample(range(len(dataset)), subset_size)
    subset_data = [dataset[i] for i in indices]
    images, labels = zip(*subset_data)
    images = torch.stack([img.view(-1) for img in images])
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(images)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(dataset.classes)))
    plt.title("t-SNE of Dataset")

    LoggingService().save_figure(plt.gcf(), "tsne", logdir)
    plt.close()
