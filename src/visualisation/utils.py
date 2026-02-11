"""Shared helper utilities for training visualisations."""

from __future__ import annotations

from typing import Any, Iterable

import logging
import numpy as np

log = logging.getLogger(__name__)

def metric_stats_from_seeds(
    seed_dict: dict[Any, Any],
    metric: str,
    group: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned step/mean/std arrays for a metric across model seeds."""

    metric_name = f"{group}/{metric}" if group else metric
    step_name = f"{group}/step" if group else "step"
    metric_values, step_values = [], []

    for seed, seed_model in seed_dict.items():
        seed_metrics = np.asarray(seed_model.metrics[metric_name])
        seed_steps = np.asarray(seed_model.metrics[step_name])
        if seed_metrics is None:
            log.error(f"Missing '{metric_name}' column for seed {seed}")
            continue
        if seed_steps is None:
            log.error(f"Missing '{step_name}' column for seed {seed}")
            continue
        if seed_steps.shape != seed_metrics.shape:
            log.error(f"Metric '{metric_name}' and steps lengths do not match for seed {seed}")
            continue

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
