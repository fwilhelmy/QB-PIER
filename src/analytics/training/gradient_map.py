"""Gradient map figure generation helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities import rank_zero_only

from analytics.utils import metric_stats_from_seeds

log = logging.getLogger(__name__)


def _gaussian_kernel(kernel_size: int = 9, sigma: float = 1.8) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel for separable smoothing."""
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _smooth_matrix(data: np.ndarray, kernel_size: int = 9, sigma: float = 1.8) -> np.ndarray:
    """Apply light Gaussian smoothing across both matrix axes."""
    row_kernel_size = min(kernel_size, data.shape[1] if data.shape[1] % 2 == 1 else data.shape[1] - 1)
    col_kernel_size = min(kernel_size, data.shape[0] if data.shape[0] % 2 == 1 else data.shape[0] - 1)

    row_kernel = _gaussian_kernel(kernel_size=row_kernel_size, sigma=sigma)
    col_kernel = _gaussian_kernel(kernel_size=col_kernel_size, sigma=sigma)

    smoothed = np.apply_along_axis(lambda row: np.convolve(row, row_kernel, mode="same"), axis=1, arr=data)
    smoothed = np.apply_along_axis(lambda col: np.convolve(col, col_kernel, mode="same"), axis=0, arr=smoothed)
    return smoothed


def plot_gradient_map(grad_dict, title):
    """Plot a smoothed gradient-strength map plus layer-wise summary bars.

    Fixes:
      - If only a single gradient sample exists (matrix has 1 column), duplicate the
        column so contourf receives at least a 2-column grid.
      - Invert layer order (last layer shown at the top), matching common gradient-flow plots.
    """
    # Keep insertion order of dict (Py3.7+) but reverse it to invert layer order.
    keys = list(grad_dict.keys())[::-1]

    layers = [k.replace(".weight", "").replace(".bias", "") for k in keys]
    raw_data = np.array([grad_dict[k] for k in keys], dtype=float)

    # Contour plots require at least a (2, 2) grid. If we only have one time sample,
    # duplicate the column to get shape (n_layers, 2).
    if raw_data.ndim != 2:
        raise ValueError(f"Expected 2D gradient matrix, got shape {raw_data.shape}")
    if raw_data.shape[1] == 1:
        raw_data = np.repeat(raw_data, 2, axis=1)
    if raw_data.shape[0] < 2:
        raise ValueError(f"Need at least 2 layers for gradient map, got {raw_data.shape[0]} layer(s)")

    num_ticks = raw_data.shape[1]
    map_data = _smooth_matrix(raw_data)

    means = [float(np.mean(grad_dict[k])) for k in keys]
    stds = [float(np.std(grad_dict[k])) for k in keys]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, len(layers) * 0.45 + 3), gridspec_kw={"width_ratios": [70, 30]}
    )

    # --- Map ---
    ax0 = axes[0]
    x = np.arange(num_ticks)
    y = np.arange(len(layers))

    contour = ax0.contourf(x, y, map_data, levels=40, cmap="viridis")
    ax0.contour(x, y, map_data, levels=10, colors="white", linewidths=0.2, alpha=0.35)

    strongest = np.unravel_index(np.argmax(map_data), map_data.shape)
    ax0.scatter(
        strongest[1],
        strongest[0],
        marker="*",
        s=180,
        color="gold",
        edgecolors="black",
        linewidths=0.8,
    )

    tick_step = max(1, num_ticks // 10)
    ax0.set_xticks(np.arange(0, num_ticks, tick_step))
    ax0.set_xticklabels([str(i + 1) for i in range(0, num_ticks, tick_step)])

    ax0.set_yticks(y)
    ax0.set_yticklabels(layers)
    ax0.yaxis.tick_left()
    ax0.yaxis.set_label_position("left")

    ax0.set_title("Smoothed Gradient Map")
    ax0.set_xlabel("Training Step")

    # --- Summary bars ---
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
def generate_gradient_map_plots(seed_dict: dict[Any, Any], fig_dir=None):
    """Generate and save smoothed gradient map visualizations."""
    from analytics.metrics_io import resolve_metrics

    if not seed_dict:
        raise ValueError("No seeds provided for gradient map plotting.")

    first_seed = next(iter(seed_dict.values()))
    grad_norms = resolve_metrics(first_seed)["grad_l2"]
    if grad_norms is None:
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

    weight_title = "Gradient Map of Weights"
    try:
        weight_fig, _ = plot_gradient_map(weight_grads, weight_title)
    except ValueError as e:
        log.warning(f"Could not generate weight gradient map: {e}")
        return
    output_dir = fig_dir or "logs"
    os.makedirs(output_dir, exist_ok=True)
    weight_fig.savefig(os.path.join(output_dir, "gradient_map_weight.png"))
    plt.close(weight_fig)

    bias_grads = {
        k: v
        for k, v in mean_grad_norms.items()
        if "bias" in k and "bn" not in k and "norm" not in k
    }
    if bias_grads == {}:
        log.info("No bias gradients found.")
        return

    bias_title = "Gradient Map of Biases"
    try:
        bias_fig, _ = plot_gradient_map(bias_grads, bias_title)
    except ValueError as e:
        log.warning(f"Could not generate bias gradient map: {e}")
        return
    os.makedirs(output_dir, exist_ok=True)
    bias_fig.savefig(os.path.join(output_dir, "gradient_map_bias.png"))
    plt.close(bias_fig)



def try_plot_gradient_map(seed_dict: dict[Any, Any], fig_dir=None):
    try:
        return generate_gradient_map_plots(seed_dict, fig_dir)
    except Exception as e:
        log.warning(f"Could not generate gradient map plot for {fig_dir}: {e}")
        return None, None
