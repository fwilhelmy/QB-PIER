"""Gradient flow figure generation helpers."""

from __future__ import annotations

import logging

from lightning.pytorch.utilities import rank_zero_only

from LoggingService import LoggingService
from .gradient_flow import plot_gradient_flow

__all__ = ["generate_gradient_flow_plots"]

log = logging.getLogger(__name__)


@rank_zero_only
def generate_gradient_flow_plots(model, fig_dir=None):
    if model.metrics["grad_l2"] == []:
        log.warning("No gradient data found in model metrics. Skipping gradient flow plots.")
        return
    grad_norms = model.metrics["grad_l2"]

    weight_grads = {
        k: v
        for k, v in grad_norms.items()
        if "weight" in k and "bias" not in k and "bn" not in k and "norm" not in k
    }
    if weight_grads == {}:
        raise ValueError("No weight gradients found.")
    weight_title = "Gradient Flow of Weights"
    weight_fig, _ = plot_gradient_flow(weight_grads, weight_title)
    LoggingService().save_figure(weight_fig, "gradient_flow_weight.png", fig_dir)

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
    LoggingService().save_figure(bias_fig, "gradient_flow_bias", fig_dir)
