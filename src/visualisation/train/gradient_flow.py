"""Gradient flow visualisations."""

from __future__ import annotations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ["plot_gradient_flow"]


def plot_gradient_flow(grad_dict, title):
    layers = [k.replace(".weight", "").replace(".bias", "") for k in grad_dict.keys()]
    num_ticks = len(next(iter(grad_dict.values())))
    data = np.array([grad_dict[k] for k in grad_dict.keys()])

    means = [np.mean(grad_dict[k]) for k in grad_dict.keys()]
    stds = [np.std(grad_dict[k]) for k in grad_dict.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(16, len(layers) * 0.4 + 3), gridspec_kw={"width_ratios": [70, 30]})

    ax0 = axes[0]
    sns.heatmap(data, ax=ax0, xticklabels=range(1, num_ticks + 1), yticklabels=layers, cmap="viridis", cbar=False)
    ax0.set_title("Gradient Flow Heatmap")
    ax0.yaxis.tick_left()
    ax0.yaxis.set_label_position("left")

    ax1 = axes[1]
    y_positions = np.arange(len(layers))
    ax1.barh(y_positions, means, xerr=stds, capsize=5, color="skyblue")
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlabel("Gradient Norm")
    ax1.set_title("Layer-wise Gradient Flow")

    bar_height = 0.8
    ax1.set_ylim((len(layers) - 1) + bar_height / 2, 0 - bar_height / 2)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes
