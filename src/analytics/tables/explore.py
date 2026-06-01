"""Print a tree of experiment directories until ``.model`` leaves are found."""

from __future__ import annotations

from pathlib import Path


def explore_dir(root: Path, indent: int = 0) -> None:
    """Recursively explore directories until a directory containing ``.model`` is found."""
    if not root.is_dir():
        return

    prefix = "│   " * indent + "├── "
    print(f"{prefix}{root.name}/")

    if any(p.name == ".model" for p in root.iterdir()):
        return

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for subdir in subdirs:
        explore_dir(subdir, indent + 1)
