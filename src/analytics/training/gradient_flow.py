"""Gradient flow figure generation helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only

from analytics.utils import metric_stats_from_seeds

log = logging.getLogger(__name__)

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

@rank_zero_only
def generate_gradient_flow_plots(seed_dict: dict[Any, Any], fig_dir=None):
    from analytics.metrics_io import resolve_metrics

    if not seed_dict:
        raise ValueError("No seeds provided for gradient flow plotting.")

    first_seed = next(iter(seed_dict.values()))
    grad_norms = resolve_metrics(first_seed)["grad_l2"]
    if grad_norms is None or grad_norms == {}:
        raise ValueError("No gradient norms found in metrics for this model.")

    mean_grad_norms: dict[str, np.ndarray] = {}
    for layer_name in grad_norms:
        _, mean_values, _ = metric_stats_from_seeds(seed_dict, metric=layer_name, group="grad_l2", x_axis="epoch")
        mean_grad_norms[layer_name] = mean_values

    weight_grads = {
        k: v
        for k, v in mean_grad_norms.items()
        if "weight" in k and "bias" not in k and "bn" not in k and "norm" not in k
    }
    if weight_grads == {}:
        raise ValueError("No weight gradients found.")
    weight_title = "Gradient Flow of Weights"
    weight_fig, _ = plot_gradient_flow(weight_grads, weight_title)
    output_dir = fig_dir or "logs"
    os.makedirs(output_dir, exist_ok=True)
    weight_filename = "gradient_flow_weight.png"
    weight_fig.savefig(os.path.join(output_dir, weight_filename))
    plt.close(weight_fig)

    bias_grads = {
        k: v
        for k, v in mean_grad_norms.items()
        if "bias" in k and "bn" not in k and "norm" not in k
    }
    if bias_grads == {}:
        log.info("No bias gradients found.")
        return
    bias_title = "Gradient Flow of Biases"
    bias_fig, _ = plot_gradient_flow(bias_grads, bias_title)
    output_dir = fig_dir or "logs"
    os.makedirs(output_dir, exist_ok=True)
    bias_filename = "gradient_flow_bias.png"
    bias_fig.savefig(os.path.join(output_dir, bias_filename))
    plt.close(bias_fig)

def try_plot_gradient_flow(seed_dict: dict[Any, Any], fig_dir=None):
    try:
        return generate_gradient_flow_plots(seed_dict, fig_dir)
    except ValueError as e:
        log.warning(f"Could not generate gradient flow plot for {fig_dir}: {e}")
        return None, None