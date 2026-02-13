"""Shared helper utilities for training visualisations."""

from __future__ import annotations

from typing import Any, Iterable
from pathlib import Path

import logging
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log = logging.getLogger(__name__)

def _extract_scalar_series(tb_dir: Path, scalar_tag: str) -> tuple[list[float], list[float]]:
    events = load_scalar_events(tb_dir=tb_dir, scalar_tag=scalar_tag)

    first_wall_time = events[0].wall_time
    elapsed_seconds = [event.wall_time - first_wall_time for event in events]
    values = [event.value for event in events]
    return elapsed_seconds, values

def load_scalar_events(tb_dir: Path, scalar_tag: str, size_guidance: int = 0):
    """Return TensorBoard scalar events for a tag from one run directory."""
    accumulator = EventAccumulator(str(tb_dir), size_guidance={EventAccumulator.SCALARS: size_guidance})
    accumulator.Reload()

    if scalar_tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"Missing scalar tag '{scalar_tag}' in TensorBoard logs at {tb_dir}")

    events = accumulator.Scalars(scalar_tag)
    if not events:
        raise ValueError(f"Scalar tag '{scalar_tag}' has no events in TensorBoard logs at {tb_dir}")

    return events

def metric_stats_from_seeds(
    seed_dict: dict[Any, Any],
    metric: str,
    group: str | None = None,
    x_axis: str = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned step/mean/std arrays for a metric across model seeds."""
    valid_x_axis = {"step", "epoch", "indice", "time"}
    if x_axis not in valid_x_axis:
        raise ValueError(f"Invalid x_axis '{x_axis}'. Expected one of: {sorted(valid_x_axis)}")
    
    metric_name = f"{group}/{metric}" if group else metric
    metric_values, step_values = [], []

    for seed, seed_model in seed_dict.items():
        if x_axis == "time":
            tb_dir = Path(seed_model.log_dir) / "tensorboard"
            if not tb_dir.is_dir():
                log.error(f"Missing tensorboard directory for seed {seed} at expected location: {tb_dir}")
                continue

            seed_steps, seed_metrics = _extract_scalar_series(tb_dir=tb_dir, scalar_tag=metric_name)
        else:
            seed_metrics = np.asarray(seed_model.metrics[metric_name])
            if seed_metrics is None:
                log.error(f"Missing '{metric_name}' column for seed {seed}")
                continue

            step_name = f"{group}/step" if group else "epoch"
            seed_steps = np.asarray(seed_model.metrics[step_name]) if x_axis == "step" else np.arange(seed_metrics.shape[0])

        if seed_steps.shape != seed_metrics.shape:
            raise ValueError(f"Inconsistent step and metric counts for seed {seed}: steps shape {seed_steps.shape}, metrics shape {seed_metrics.shape}")

        step_values.append(seed_steps)
        metric_values.append(seed_metrics)

    if not step_values:
        raise ValueError(f"No valid '{metric_name}' metrics found across seeds")

    steps = step_values[0]
    if any(seed_steps.shape != steps.shape for seed_steps in step_values):
        raise ValueError(f"Inconsistent step counts across seeds for metric '{metric_name}'")

    metric_stack = np.vstack(metric_values)
    return steps, metric_stack.mean(axis=0), metric_stack.std(axis=0)


def line_with_std(
    ax,
    x: Iterable[float] | np.ndarray | None = None,
    mean: Iterable[float] | np.ndarray | None = None,
    std: Iterable[float] | np.ndarray | None = None,
    label: str | None = None,
    **plot_kwargs: Any,
) -> None:
    """Plot a line with a shaded standard deviation band.

    Either provide explicit ``x``, ``mean`` and ``std`` arrays, or provide
    ``seed_dict`` with a ``metric_name`` to compute statistics from seed
    MetricContainers.
    """
    ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(x, np.asarray(mean) - np.asarray(std), np.asarray(mean) + np.asarray(std), alpha=0.2)
