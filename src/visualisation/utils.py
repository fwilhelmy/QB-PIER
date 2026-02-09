"""Shared helper utilities for training visualisations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = ["line_with_std"]


def line_with_std(
    ax,
    x: Iterable[float] | np.ndarray,
    mean: Iterable[float] | np.ndarray,
    std: Iterable[float] | np.ndarray,
    label: str | None = None,
    **plot_kwargs: Any,
) -> None:
    """Plot a line with a shaded standard deviation band."""

    ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(x, np.asarray(mean) - np.asarray(std), np.asarray(mean) + np.asarray(std), alpha=0.2)
