"""Regenerate transfer attack plots from a CSV file.

This utility reads the ``results.csv`` produced by ``loading.py`` and
recreates the transfer attack figures without re-running the attacks.

Usage
-----
python plot_transfer_from_csv.py <root_dir>
    <root_dir>  Directory passed to ``loading.py`` where ``attacks/results.csv``
                is stored.
"""

from __future__ import annotations

import os
import sys
import pandas as pd

from metrics.LoggingService import LoggingService
from visualisation.attacks import plot_transfer_attack_effects


def main(root_dir: str) -> None:
    """Load attack results from ``root_dir`` and regenerate plots."""
    ls = LoggingService(root_dir)
    csv_path = os.path.join(ls.root_dir, "attacks", "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find attack results at '{csv_path}'")

    df = pd.read_csv(csv_path)
    plot_transfer_attack_effects(df, title_prefix="attacks/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_transfer_from_csv.py <root_dir>")
        sys.exit(1)
    main(sys.argv[1])
