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


def _is_metrics_container(obj: object) -> bool:
    """True for a loaded metrics container, not a seed dict or mistaken empty lookup."""
    return (
        not isinstance(obj, dict)
        and hasattr(obj, "log_dir")
        and hasattr(obj, "__getitem__")
    )


def resolve_metrics(seed_obj: MetricsContainerLike | SeedModelLike) -> MetricsContainerLike:
    """Return an indexable metrics container regardless of seed object type."""
    if isinstance(seed_obj, dict):
        raise TypeError(
            "Expected a per-seed MetricsContainer or model instance, not a seed dict "
            f"(keys: {list(seed_obj.keys())[:5]}...)"
        )

    # BaseModel and similar: real ``.metrics`` attribute pointing at MetricsContainer.
    inner = getattr(seed_obj, "metrics", None)
    if _is_metrics_container(inner):
        return inner  # type: ignore[return-value]

    # Bare MetricsContainer from ``load_metrics`` (no wrapper model).
    if _is_metrics_container(seed_obj):
        return seed_obj  # type: ignore[return-value]

    raise TypeError(f"Seed object is not metrics-like: {type(seed_obj).__name__}")

