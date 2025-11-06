"""Attack result persistence helpers."""

from __future__ import annotations

import os

import pandas as pd
from lightning.pytorch.utilities import rank_zero_only

__all__ = ["store_attack_results"]


@rank_zero_only
def store_attack_results(records: list, root_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    os.makedirs(root_dir, exist_ok=True)
    out_path = os.path.join(root_dir, "attack_results.csv")
    df.to_csv(out_path, index=False)
    return df
