"""Shared helper utilities for training visualisations."""

from __future__ import annotations

from typing import Any, Iterable
from pathlib import Path

import logging
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log = logging.getLogger(__name__)


def find_tensorboard_dir(run_dir: Path) -> Path:
    """Locate the TensorBoard directory under a run path."""
    direct = run_dir / "tensorboard"
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"No tensorboard directory found under: {run_dir}")


def _extract_scalar_series(tb_dir: Path, scalar_tag: str) -> tuple[list[float], list[float]]:
    accumulator = EventAccumulator(str(tb_dir))
    accumulator.Reload()

    if scalar_tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"Missing scalar tag '{scalar_tag}' in TensorBoard logs at {tb_dir}")

    events = accumulator.Scalars(scalar_tag)
    if not events:
        raise ValueError(f"Scalar tag '{scalar_tag}' has no events in TensorBoard logs at {tb_dir}")

    first_wall_time = events[0].wall_time
    elapsed_seconds = [event.wall_time - first_wall_time for event in events]
    values = [event.value for event in events]
    return np.asarray(elapsed_seconds), np.asarray(values)

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
            try:
                tb_dir = find_tensorboard_dir(Path(seed_model.log_dir))
                seed_steps, seed_metrics = _extract_scalar_series(tb_dir=tb_dir, scalar_tag=metric_name)
            except FileNotFoundError as e:
                log.warning(f"Could not load metrics from tensorboard for seed {seed}: {e}")
                continue
        else:
            seed_metrics = np.asarray(seed_model.metrics[metric_name])
            if seed_metrics.size == 0 or seed_metrics.ndim != 1:
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

def format_x_axis(ax, epochs, *, bottom="step", top="epoch", nbins=8):
    """
    Unbiased x-axis formatter.

    Curves remain plotted against STEP (true x). Tick positions remain in step-space.
    Epoch is displayed only via a label mapping step -> epoch using `epochs`.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis with data plotted vs step.
    epochs : array-like
        epochs[i] is the step where epoch i+1 ended.
    bottom : {"step","epoch"}
        What to display on the bottom axis.
    top : {"step","epoch",None}
        What to display on the top axis. If same as bottom, top axis is omitted.
    nbins : int
        Tick density for both axes (MaxNLocator in step-space).
    """
    epochs = np.asarray(epochs, dtype=float)
    epochs = np.unique(epochs)
    if epochs.size == 0:
        raise ValueError("`epochs` must be a non-empty array of epoch-end steps.")
    epoch_numbers = np.arange(1, len(epochs) + 1)

    def epoch_fmt(x, pos):
        idx = np.searchsorted(epochs, x, side="right") - 1
        if 0 <= idx < len(epoch_numbers):
            return str(int(epoch_numbers[idx]))
        return ""

    def step_fmt(x, pos):
        return str(int(x))

    def apply(axis, mode):
        axis.xaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=True))
        axis.xaxis.set_major_formatter(FuncFormatter(epoch_fmt if mode == "epoch" else step_fmt))

        # Keep ticks within the epoch span and include endpoints (still step-space positions)
        ticks = np.asarray(axis.get_xticks(), dtype=float)
        ticks = ticks[(ticks >= epochs[0]) & (ticks <= epochs[-1])]
        ticks = np.unique(np.concatenate([ticks, [epochs[0], epochs[-1]]]))
        axis.set_xticks(ticks)

    # Bottom (real axis)
    apply(ax, "epoch" if bottom == "epoch" else "step")

    # Top (optional)
    ax_top = None
    if top is not None and top != bottom:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        apply(ax_top, "epoch" if top == "epoch" else "step")
        ax_top.xaxis.set_label_position("top")
        ax_top.xaxis.tick_top()

    return ax_top