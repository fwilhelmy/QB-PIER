import json
import logging
import os
import pickle
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import psutil
import torch
from pytorch_lightning.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only

log = logging.getLogger(__name__)


class MetricsContainer(Logger):
    def __init__(
        self,
        log_dir: str,
    ):
        super().__init__()
        self._raw: Dict[str, Any] = {}
        self._last_logged_step: Dict[str, Optional[int]] = {}
        self.log_dir = os.path.join(log_dir, "metrics")
        self._init_directories()

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def version(self) -> str:
        return "0"

    @property
    def experiment(self) -> "MetricsContainer":
        return self

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        # Hyperparameters are tracked elsewhere; no-op to satisfy the logger interface.
        return

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
        if not metrics:
            return

        grouped_steps: Dict[str, int] = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.detach().cpu().tolist()
            elif isinstance(value, (torch.Tensor,)):
                value = value.detach().cpu().tolist()

            group: Optional[str] = None
            metric_name = key
            if "/" in key:
                group, metric_name = key.split("/", 1)

            self.log_scalar(metric_name, value, group=group)

            if group is not None and step is not None:
                grouped_steps.setdefault(group, step)

        for group, group_step in grouped_steps.items():
            if self._last_logged_step.get(group) != group_step:
                self._raw.setdefault(group, {}).setdefault("step", []).append(group_step)
                self._last_logged_step[group] = group_step

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
        df = self.to_pandas()
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
    def to_pandas(self) -> pd.DataFrame:
        """Compile the logged metrics into a pandas DataFrame indexed by epoch.

        All metric groups present in the container are converted to DataFrames, aligned on the
        epoch column and merged together. Validation sanity-check rows are kept at the top of the
        resulting DataFrame, mirroring the behaviour of the previous :meth:`format_metrics`.
        """

        def _assign_epochs(steps: pd.Series, epoch_steps: List[int]) -> List[int]:
            epochs, bi = [], 0
            n_epochs = len(epoch_steps)
            for step in steps:
                while bi < n_epochs and step > epoch_steps[bi]:
                    bi += 1
                epochs.append(bi)
            return epochs

        epoch_steps = self._raw.get("epoch", [])
        if not epoch_steps:
            return pd.DataFrame()

        frames = []
        sanity_row: Optional[pd.DataFrame] = None

        for group, metrics in self._raw.items():
            if group == "epoch" or not isinstance(metrics, dict):
                continue
            df = pd.DataFrame(metrics)
            if df.empty:
                continue
            if "step" not in df.columns:
                continue

            df = df.sort_values("step").reset_index(drop=True)
            df.insert(0, "epoch", _assign_epochs(df["step"], epoch_steps))

            prefix = f"{group}_" if group else ""
            rename_map = {}
            for col in df.columns:
                if col == "epoch":
                    continue
                if col == "step":
                    if group == "train":
                        rename_map[col] = "step"
                    else:
                        rename_map[col] = f"{prefix}step" if prefix else col
                else:
                    rename_map[col] = f"{prefix}{col}"
            df = df.rename(columns=rename_map)

            if group == "val":
                sanity_row = df.iloc[[0]].copy()
                df = df.iloc[1:].reset_index(drop=True)

            frames.append(df)

        if not frames:
            return sanity_row.reset_index(drop=True) if sanity_row is not None else pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = pd.merge(merged, frame, on="epoch", how="outer", sort=True)

        merged = merged.sort_values("epoch").reset_index(drop=True)

        if sanity_row is not None:
            merged = pd.concat([sanity_row, merged], ignore_index=True, sort=False)

        return merged.reset_index(drop=True)

