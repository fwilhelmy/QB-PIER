"""Gradient map figure generation helpers."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities import rank_zero_only

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
    """Plot a smoothed gradient-strength map plus layer-wise summary bars."""
    layers = [k.replace(".weight", "").replace(".bias", "") for k in grad_dict.keys()]
    num_ticks = len(next(iter(grad_dict.values())))
    raw_data = np.array([grad_dict[k] for k in grad_dict.keys()], dtype=float)
    map_data = _smooth_matrix(raw_data)

    means = [np.mean(grad_dict[k]) for k in grad_dict.keys()]
    stds = [np.std(grad_dict[k]) for k in grad_dict.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(16, len(layers) * 0.45 + 3), gridspec_kw={"width_ratios": [70, 30]})

    ax0 = axes[0]
    x = np.arange(num_ticks)
    y = np.arange(len(layers))
    contour = ax0.contourf(x, y, map_data, levels=40, cmap="viridis")
    ax0.contour(x, y, map_data, levels=10, colors="white", linewidths=0.2, alpha=0.35)

    strongest = np.unravel_index(np.argmax(map_data), map_data.shape)
    ax0.scatter(strongest[1], strongest[0], marker="*", s=180, color="gold", edgecolors="black", linewidths=0.8)

    cbar = fig.colorbar(contour, ax=ax0, fraction=0.046, pad=0.03)
    cbar.set_label("Smoothed Gradient Strength")

    tick_step = max(1, num_ticks // 10)
    ax0.set_xticks(np.arange(0, num_ticks, tick_step))
    ax0.set_xticklabels([str(i + 1) for i in range(0, num_ticks, tick_step)])
    ax0.set_yticks(y)
    ax0.set_yticklabels(layers)
    ax0.set_title("Smoothed Gradient Map")
    ax0.set_xlabel("Training Step")
    ax0.set_ylabel("Layer")

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
def generate_gradient_flow_plots(model, fig_dir=None):
    """Generate and save smoothed gradient map visualizations."""
    try:
        grad_norms = model.metrics["grad_l2"]
    except AttributeError:
        log.warning("No gradient norms found in model metrics; skipping gradient map plots.")
        return

    weight_grads = {
        k: v
        for k, v in grad_norms.items()
        if "weight" in k and "bias" not in k and "bn" not in k and "norm" not in k
    }
    if weight_grads == {}:
        raise ValueError("No weight gradients found.")

    weight_title = "Gradient Map of Weights"
    weight_fig, _ = plot_gradient_map(weight_grads, weight_title)
    output_dir = fig_dir or "logs"
    os.makedirs(output_dir, exist_ok=True)
    weight_fig.savefig(os.path.join(output_dir, "gradient_map_weight.png"))
    plt.close(weight_fig)

    bias_grads = {
        k: v
        for k, v in grad_norms.items()
        if "bias" in k and "bn" not in k and "norm" not in k
    }
    if bias_grads == {}:
        log.info("No bias gradients found.")
        return

    bias_title = "Gradient Map of Biases"
    bias_fig, _ = plot_gradient_map(bias_grads, bias_title)
    os.makedirs(output_dir, exist_ok=True)
    bias_fig.savefig(os.path.join(output_dir, "gradient_map_bias.png"))
    plt.close(bias_fig)



def generate_gradient_map_plots(model, fig_dir=None):
    """Backward-compatible alias for gradient map generation."""
    return generate_gradient_flow_plots(model, fig_dir=fig_dir)
