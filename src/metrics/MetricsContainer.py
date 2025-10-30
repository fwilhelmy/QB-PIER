import json
import logging
import os
import pickle
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import pandas as pd
import psutil
import torch
from lightning.pytorch.utilities import rank_zero_only

log = logging.getLogger(__name__)

class MetricsContainer():
    def __init__(
        self,
        log_dir: str,
    ):
        self._raw: Dict[str, Any] = {}
        self.log_dir = os.path.join(log_dir, "metrics")

    @rank_zero_only
    def _init_directories(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def __getitem__(self, key: str):
        if "/" in key:
            group, metric = key.split("/", 1)
            return self._raw.setdefault(group, {}).get(metric, {})
        return self._raw.get(key, [])
    
    def log_scalar(self, metric: str, value: Any, group: Optional[str] = None) -> None:
        bucket = self._raw.setdefault(group, {}) if group is not None else self._raw
        bucket.setdefault(metric, []).append(value)

    def log_dict(self, group: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self.log_scalar(key, value, group=group)
        if step is not None:
            self._raw[group].setdefault("step", []).append(step)

    @rank_zero_only
    def log_epoch(self, step: int) -> None:
        self.log_scalar("epoch", step)

    def to_csv(self, path: Optional[str] = None) -> None:
        df = self.format_metrics()
        if df is None or df.empty:
            return
        out = path or os.path.join(self.log_dir, "metrics.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)

    def to_json(self, path: Optional[str] = None) -> None:
        out = path or os.path.join(self.log_dir, "metrics.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._raw, f, indent=2)

    @rank_zero_only
    def save_metrics(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "logger.pkl"), "wb") as f:
            pickle.dump({
                "_raw": self._raw
            }, f)
        self.to_csv()

    @rank_zero_only
    def load_metrics(self, log_dir: str) -> None:
        pkl_path = os.path.join(log_dir, "metrics", "logger.pkl")
        try:
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Metrics file not found at {pkl_path}")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self._raw = data.get("_raw", {})
        except Exception as err:
            log.exception(RuntimeError(f"Error loading metrics: {err}"))
        
    @rank_zero_only
    def format_metrics(self) -> pd.DataFrame:
        """
        Formats and merges training and validation metrics into a single pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing merged and renamed columns for training and validation metrics,
            with an 'epoch' column and a sanity-check row at the top. If no training data is present, returns an empty DataFrame.
        """
        train = pd.DataFrame(self._raw.get("train", {}))
        if train.empty:
            return pd.DataFrame()
        train = train.sort_values("step").reset_index(drop=True)
        train = train.rename(columns={c: f"train_{c}" for c in train.columns if c != "step"})
        
        val = pd.DataFrame(self._raw.get("val", {}))
        if val.empty:
            return train
        val = val.sort_values("step").reset_index(drop=True)
        val = val.rename(columns={c: f"val_{c}" for c in val.columns if c != "step"})

        # Sanity check first row
        sanity_row = val.iloc[[0]].copy()
        val = val.iloc[1:].reset_index(drop=True)

        # Merging train and val dataframes
        merged = pd.merge_asof(train, val, on="step", direction="nearest", tolerance=1)

        # Assign epochs by walking through epoch-step bindings
        n_epochs = len(self._raw['epoch'])
        epochs, bi = [], 0
        for step in merged["step"]:
            while step > self._raw['epoch'][bi] and bi < n_epochs:
                bi += 1
            epochs.append(bi)
        merged.insert(0, "epoch", epochs)

        # Re-insert the sanity-check row at the top (epoch 0)
        sanity_row.insert(0, "epoch", 0)
        merged = pd.concat([sanity_row, merged], ignore_index=True, sort=False)

        return merged.reset_index(drop=True)

