"""Shared path helpers for analytics notebooks and CLIs."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or cwd) until a directory containing ``src`` is found."""
    current = (start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "src").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (no parent directory contains 'src').")


def default_log_root(project_root: Path) -> Path:
    return training_log_root(project_root)


def training_log_root(project_root: Path) -> Path:
    return project_root / "logs" / "ByModel"


def attack_log_root(project_root: Path) -> Path:
    return project_root / "logs" / "ATTACKS"


def attack_experiment_dir(project_root: Path) -> Path:
    return attack_log_root(project_root) / "attack_experiment"


def attack_tables_dir(project_root: Path) -> Path:
    return attack_experiment_dir(project_root) / "tables"


def output_dir(project_root: Path, name: str) -> Path:
    path = project_root / "outputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
