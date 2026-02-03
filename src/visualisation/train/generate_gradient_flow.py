"""Gradient flow figure generation helpers."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only

from .gradient_flow import plot_gradient_flow

__all__ = ["generate_gradient_flow_plots"]

log = logging.getLogger(__name__)


@rank_zero_only
def generate_gradient_flow_plots(model, fig_dir=None):
    try:
        grad_norms = model.metrics["grad_l2"]
    except AttributeError:
        log.warning("No gradient norms found in model metrics; skipping gradient flow plots.")
        return

    weight_grads = {
        k: v
        for k, v in grad_norms.items()
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
        for k, v in grad_norms.items()
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
