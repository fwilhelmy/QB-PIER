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
        self._log_dir = os.path.join(log_dir, "metrics")

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def experiment(self) -> "MetricsContainer":
        return self

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @property
    def version(self) -> int | str | None:
        return None

    # Hyperparameters are tracked elsewhere; no-op to satisfy the logger interface.
    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        return

    @rank_zero_only
    def _init_directories(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def __getitem__(self, key: str):
        if key == "all":
            return self._raw

        if "/" in key:
            group, metric = key.split("/", 1)
            group_metrics = self._raw.get(group)
            if group_metrics is not None and metric in group_metrics:
                return group_metrics[metric]

        if key in self._raw:
            return self._raw[key]

        log.warning(f"Metric {key!r} not found")
        return {}

    def __getattr__(self, item: str) -> Any:
        """Allow direct attribute access to stored metric groups."""

        return self[item]
    
    def series(self, key: str) -> pd.Series:
        """Return a pandas Series for a given metric key, indexed by epoch."""
        df = self.to_pandas()
        if key in df.columns:
            return df[key]
        else:
            log.warning(f"Metric {key!r} not found in DataFrame columns")
            return pd.Series(dtype=float)
    
    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """
        Step-aware logging:
        - supports "group/metric" keys
        - logs tensors as python scalars/lists
        - appends `step` AT MOST ONCE per group per step
        """
        if not metrics:
            return

        # 1) group by prefix
        grouped: Dict[Optional[str], Dict[str, Any]] = {}

        for key, value in metrics.items():
            # normalize
            if torch.is_tensor(value):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.detach().cpu().tolist()

            group = None
            metric_name = key
            if "/" in key:
                group, metric_name = key.split("/", 1)

            grouped.setdefault(group, {})[metric_name] = value

        # 2) write to storage
        for group, grp_metrics in grouped.items():
            if group is None:
                # ungrouped → just scalars, no step column
                for m, v in grp_metrics.items():
                    self.log_scalar(m, v)
                continue

            # grouped
            if step is not None:
                last_step = self._last_logged_step.get(group)
                if last_step != step:
                    # first time we see this group at this step → log dict + step
                    self.log_dict(group, grp_metrics, step=step)
                    self._last_logged_step[group] = step
                else:
                    # same step → no extra step append
                    self.log_dict(group, grp_metrics, step=None)
            else:
                # no step provided → just values
                self.log_dict(group, grp_metrics, step=None)
    
        return None

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
                else:
                    rename_map[col] = f"{prefix}{col}" if prefix else col
            df = df.rename(columns=rename_map)

            frames.append(df)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = pd.merge(merged, frame, on="epoch", how="outer", sort=True)

        merged = merged.sort_values("epoch").reset_index(drop=True)

        return merged.reset_index(drop=True)