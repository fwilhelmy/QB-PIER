"""Shared utilities for training visualisation scripts."""

from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from visualisation.training.arch_gradient_norms import ScalarSeries


def default_run_label(run_dir: Path) -> str:
    """Return a concise default label derived from the run directory path."""
    parent = run_dir.parent.name
    return f"{parent}/{run_dir.name}" if parent else run_dir.name


def find_tensorboard_dir(run_dir: Path) -> Path:
    """Locate the TensorBoard directory under a run path.

    Supported layouts include direct ``run/tensorboard`` and nested
    ``run/<seed>/tensorboard`` structures.
    """
    candidates = [
        run_dir / "tensorboard",
        run_dir / "seed_0" / "tensorboard",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    nested = sorted(path for path in run_dir.glob("*/tensorboard") if path.is_dir())
    if nested:
        return nested[0]

    raise FileNotFoundError(f"No tensorboard directory found under: {run_dir}")

def _extract_scalar_series(tb_dir: Path, scalar_tag: str) -> ScalarSeries:
    events = load_scalar_events(tb_dir=tb_dir, scalar_tag=scalar_tag)

    first_wall_time = events[0].wall_time
    elapsed_seconds = [event.wall_time - first_wall_time for event in events]
    values = [event.value for event in events]
    return ScalarSeries(elapsed_seconds=elapsed_seconds, values=values)

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
