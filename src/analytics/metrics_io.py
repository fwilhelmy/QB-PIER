"""Lightweight adapters between seed models and metric containers.

Training plot functions need to index metrics via keys like ``"train/loss"`` and
``"val/step"``. Depending on how runs are loaded, per-seed objects can be either:

- a metrics container (from `loading.load_metrics`)
- a loaded model object exposing `.metrics` (from `loading.load_model`)

This module avoids importing heavyweight training dependencies at import-time
and instead uses duck-typing for normalization.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypeAlias


class MetricsContainerLike(Protocol):
    log_dir: str

    def __getitem__(self, key: str) -> Any: ...


class SeedModelLike(Protocol):
    metrics: MetricsContainerLike


SeedMetrics: TypeAlias = Mapping[int, MetricsContainerLike | SeedModelLike]
ModelMetrics: TypeAlias = Mapping[str, SeedMetrics]


def resolve_metrics(seed_obj: MetricsContainerLike | SeedModelLike) -> MetricsContainerLike:
    """Return an indexable metrics container regardless of seed object type."""
    metrics = getattr(seed_obj, "metrics", None)
    if metrics is not None and hasattr(metrics, "__getitem__"):
        return metrics
    if hasattr(seed_obj, "__getitem__"):
        return seed_obj  # already a metrics-like container
    raise TypeError(f"Seed object is not metrics-like: {type(seed_obj).__name__}")

