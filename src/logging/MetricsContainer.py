import os
import time
import json
import pickle
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple

import psutil
import pandas as pd
import torch
from typing_extensions import override
from lightning.pytorch.utilities import rank_zero_only
from Configuration import LOGS_DIR

class MetricsContainer():
    def __init__(
        self,
        save_dir: str = LOGS_DIR,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._raw: Dict[str, Any] = {}

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return os.path.join(self.save_dir, "metrics")

    def __getitem__(self, key: str):
        if "/" in key:
            group, metric = key.split("/", 1)
            grp = self._raw.setdefault(group, {})
            return grp.setdefault(metric, [])
        else:
            return self._raw.setdefault(key, [])
    
    def log_scalar(self, metric: str, value: Any, group: Optional[str] = None) -> None:
        bucket = self._raw.setdefault(group, {}) if group is not None else self._raw
        bucket = bucket.setdefault(metric, []).append(value)

    def log_dict(self, group: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self.log_scalar(key, value, group=group)
        if step is not None:
            self._raw[group].setdefault("step", []).append(step if step is not None else 0)

    @rank_zero_only
    def log_epoch(self, step: int) -> None:
        self.log_scalar("epoch", step)

    @contextmanager
    def timeit(self, name: str, step: Optional[int] = None):
        start = time.time()
        yield
        elapsed = time.time() - start
        self.log_scalar("timer", elapsed, group=name)
        self.log_scalar("step", step, group=name)

    def log_resources(self, step: Optional[int] = None) -> None:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used_gb = (mem.total - mem.available) / (1024 ** 3)
        metrics = {
            "cpu_percent": cpu_pct,
            "mem_used_gb": mem_used_gb
        }
        if torch.cuda.is_available():
            metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        self.log_dict("performance", metrics, step=step)

    def to_csv(self, path: Optional[str] = None) -> None:
        df = self.get_metrics()
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
    def save(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "logger.pkl"), "wb") as f:
            pickle.dump({
                "_raw": self._raw
            }, f)
        self.to_csv()

    @rank_zero_only
    def load(self, path: Optional[str] = None) -> None:
        base = path or self.log_dir
        pkl = os.path.join(base, "metrics", "logger.pkl")
        if not os.path.exists(pkl):
            return
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        self._raw = data.get("_raw", {})
        
    @rank_zero_only
    def get_metrics(self) -> pd.DataFrame:
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
        merged = pd.merge_asof(
            train,
            val,
            on="step",
            direction="nearest",
            tolerance=1
        )

        # Assign epochs by walking through epoch-step bindings
        epochs, bi = [], 0
        for step in merged["step"]:
            while step > self['epoch'][bi] and bi < len(self['epoch']):
                bi += 1
            epochs.append(bi)
        merged.insert(0, "epoch", epochs)

        # Re-insert the sanity-check row at the top (epoch 0)
        sanity_row.insert(0, "epoch", 0)
        merged = pd.concat([sanity_row, merged], ignore_index=True, sort=False)

        return merged.reset_index(drop=True)

