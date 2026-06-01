"""Bootstrap helpers for analytics Jupyter notebooks.

Notebooks must add ``src/`` to ``sys.path`` before ``import analytics`` (see the
setup cell in ``notebooks/_template.ipynb`` or run ``scripts/create_analytics_notebooks.py``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from analytics.paths import (
    attack_experiment_dir,
    default_log_root,
    find_project_root,
    output_dir,
    training_log_root,
)

log = logging.getLogger(__name__)


def setup_notebook(start: Path | None = None) -> tuple[Path, Path]:
    """Insert ``src`` on ``sys.path`` and return ``(project_root, src_dir)``."""
    project_root = find_project_root(start)
    src_dir = project_root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return project_root, src_dir


def notebook_paths(
    notebook_name: str,
    start: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return ``(project_root, training_root, attack_experiment_dir, output_dir, src_dir)``."""
    project_root, src_dir = setup_notebook(start)
    return (
        project_root,
        training_log_root(project_root),
        attack_experiment_dir(project_root),
        output_dir(project_root, notebook_name),
        src_dir,
    )


def load_all_metrics(
    model_dirs: list[Path],
    loader: Any,
) -> dict[str, Any]:
    """Load metrics for each model directory, skipping failures with a warning."""
    models: dict[str, Any] = {}
    for model_dir in model_dirs:
        rel = str(model_dir)
        try:
            models[rel] = loader(str(model_dir))
        except Exception as exc:
            log.warning("Skipping metrics for %s: %s", rel, exc)
    return models
