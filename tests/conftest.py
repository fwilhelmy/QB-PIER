"""Test configuration for QB-PIER."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root and ``src`` directory are on ``sys.path`` so imports work consistently.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
