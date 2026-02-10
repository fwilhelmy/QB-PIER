"""Shared helper utilities for training visualisations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = ["line_with_std"]


def _metric_stats_from_seeds(
    seed_dict: dict[Any, Any],
    metric_name: str,
    step_name: str = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned step/mean/std arrays for a metric across model seeds."""

    metric_values, step_values = [], []

    for seed, seed_model in seed_dict.items():
        metrics_df = seed_model.metrics.to_pandas()
        if metrics_df.empty:
            continue
        if step_name not in metrics_df.columns:
            raise ValueError(f"Missing '{step_name}' column for seed {seed}")
        if metric_name not in metrics_df.columns:
            raise ValueError(f"Missing '{metric_name}' column for seed {seed}")

        seed_steps = metrics_df[step_name].to_numpy()
        seed_metric = metrics_df[metric_name].to_numpy()

        if seed_steps.shape != seed_metric.shape:
            raise ValueError(
                f"Metric '{metric_name}' and step column '{step_name}' lengths do not match for seed {seed}"
            )

        step_values.append(seed_steps)
        metric_values.append(seed_metric)

    if not step_values:
        raise ValueError(f"No valid '{metric_name}' metrics found across seeds")

    steps = step_values[0]
    if any(seed_steps.shape != steps.shape for seed_steps in step_values):
        raise ValueError(f"Inconsistent step counts across seeds for metric '{metric_name}'")
    if any(not np.array_equal(seed_steps, steps) for seed_steps in step_values):
        raise ValueError(f"Inconsistent step values across seeds for metric '{metric_name}'")

    metric_stack = np.vstack(metric_values)
    return steps, metric_stack.mean(axis=0), metric_stack.std(axis=0)


def line_with_std(
    ax,
    x: Iterable[float] | np.ndarray | None = None,
    mean: Iterable[float] | np.ndarray | None = None,
    std: Iterable[float] | np.ndarray | None = None,
    label: str | None = None,
    seed_dict: dict[Any, Any] | None = None,
    metric_name: str | None = None,
    step_name: str = "step",
    **plot_kwargs: Any,
) -> None:
    """Plot a line with a shaded standard deviation band.

    Either provide explicit ``x``, ``mean`` and ``std`` arrays, or provide
    ``seed_dict`` with a ``metric_name`` to compute statistics from seed
    MetricContainers.
    """

    if seed_dict is not None:
        if metric_name is None:
            raise ValueError("'metric_name' is required when using 'seed_dict'")
        x, mean, std = _metric_stats_from_seeds(seed_dict, metric_name=metric_name, step_name=step_name)

    if x is None or mean is None or std is None:
        raise ValueError("Provide either (x, mean, std) or (seed_dict, metric_name)")

    ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(x, np.asarray(mean) - np.asarray(std), np.asarray(mean) + np.asarray(std), alpha=0.2)
