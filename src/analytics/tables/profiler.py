"""Parse PyTorch profiler text output into tabular form."""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def parse_profiler_txt(path: str) -> pd.DataFrame:
    """Parse a PyTorch Lightning or cProfile-like profiler text output."""
    text = Path(path).read_text(errors="ignore")
    number_row = re.compile(
        r"^\s*(\d+)(?:/(\d+))?\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(.+)$"
    )
    rows = []
    lines = text.splitlines()
    i = 0
    current_section = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("Profile stats for:"):
            current_section = line.split("Profile stats for:", 1)[1].strip()
            i += 1
            while i < len(lines) and "ncalls" not in lines[i]:
                i += 1
            i += 1
            while i < len(lines):
                row_line = lines[i]
                match = number_row.match(row_line)
                if not match:
                    break
                ncalls, ncalls_denom, tottime, cumtime, percall, percall_cum, func = match.groups()
                rows.append(
                    {
                        "section": current_section,
                        "ncalls": int(ncalls),
                        "tottime": float(tottime),
                        "cumtime": float(cumtime),
                        "percall": float(percall),
                        "function": func.strip(),
                    }
                )
                i += 1
            continue
        i += 1

    return pd.DataFrame(rows)


def print_filtered_summary(
    df: pd.DataFrame,
    min_cumtime: float = 0.0,
    top_n: int = 20,
) -> None:
    """Print a quick summary of the slowest profiler entries."""
    if df.empty:
        log.warning("Profiler dataframe is empty.")
        return
    filtered = df[df["cumtime"] >= min_cumtime].sort_values("cumtime", ascending=False)
    log.info("\n%s", filtered.head(top_n).to_string(index=False))
