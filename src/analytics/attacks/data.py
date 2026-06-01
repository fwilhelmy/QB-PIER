"""Helpers for working with stored attack aggregate results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.paths import attack_experiment_dir, find_project_root
from loading import load_attack_aggregates


def load_attack_results(project_root: Path | None = None) -> pd.DataFrame:
    """Load attack aggregates from `logs/ATTACK/attack_experiment/tables/`."""
    root = project_root or find_project_root()
    return load_attack_aggregates(attack_experiment_dir=attack_experiment_dir(root))

