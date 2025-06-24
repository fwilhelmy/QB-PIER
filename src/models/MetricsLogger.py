import os
import pickle
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from typing_extensions import override
from pytorch_lightning.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from ..Configuration import LOGS_DIR


class MetricsLogger(Logger):
    """
    A Lightning Logger that dynamically captures all self.log(...) or self.log_dict(...)
    calls in-memory by phase and supports epoch-level boundaries.

    - log_metrics(...) inside steps for batch or on_epoch metrics.
    - @rank_zero_only log_epoch(epoch_idx, step) once per epoch to mark boundaries.
    - save() writes raw state and CSV under log_dir/metrics/.
    - load_metrics() restores previous state from log_dir.
    - get_metrics() returns a DataFrame merging train & val on global steps, with epoch assignment.
    - __getitem__ allows direct access to phase buckets via logger['train'] etc.
    """

    def __init__(
        self,
        save_dir: Optional[str] = LOGS_DIR,
        name: Optional[str] = "MetricsLogger",
        version: Optional[str | int] = None,
    ):
        super().__init__()
        # store config
        self._save_dir = save_dir
        self._logger_name = name
        self._version = version
        # dynamic phase->metric->values
        self.all_metrics: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        # epoch boundaries: list of (epoch_idx, global_step)
        self.epoch_records: List[Tuple[int, int]] = []

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        return self._logger_name

    @property
    def version(self) -> Optional[str | int]:
        return self._version

    @property
    def log_dir(self) -> str:
        base = self.save_dir or LOGS_DIR
        return os.path.join(base, "metrics")

    def __getitem__(self, phase: str) -> Dict[str, List[Any]]:
        """
        Allow dict-like access: logger['train'] returns the train metrics bucket.
        """
        return self.all_metrics.get(phase, {})

    @override
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Called by Lightning on self.log(...) or self.log_dict(...).
        'step' is Trainer.global_step.
        Records each metric under its phase prefix or 'global'.
        """
        global_step = step if step is not None else 0
        for key, value in metrics.items():
            if "/" in key:
                phase, name = key.split("/", 1)
            else:
                phase, name = "global", key
            bucket = self.all_metrics[phase]
            bucket[name].append(value)
            bucket.setdefault("step", []).append(global_step)

    @rank_zero_only
    def log_epoch(self, epoch_idx: int, step: int) -> None:
        """
        Record an epoch boundary once per epoch.
        """
        self.epoch_records.append((epoch_idx, step))

    @override
    @rank_zero_only
    def save(self) -> None:
        """
        Serialize to raw pickle and CSV under log_dir/metrics/.
        """
        os.makedirs(self.log_dir, exist_ok=True)

        # Raw logger state
        raw_path = os.path.join(self.log_dir, "logger.pkl")
        with open(raw_path, "wb") as f:
            pickle.dump({
                "all_metrics": dict(self.all_metrics),
                "epoch_records": self.epoch_records
            }, f)

        # CSV
        df = self.get_metrics()
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        df.to_csv(csv_path, index=False)

    @override
    @rank_zero_only
    def load(self, path=None) -> None:
        file = os.path.join(path or self.log_dir, "metrics", "logger.pkl")
        if not os.path.exists(file):
            return
        with open(file, "rb") as f:
            data = pickle.load(f)
        self.all_metrics = defaultdict(lambda: defaultdict(list), data.get("all_metrics", {}))
        self.epoch_records = data.get("epoch_records", [])

    @rank_zero_only
    def get_metrics(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per train log entry:
        columns: ['epoch','step', 'train_<m>...', 'val_<m>...']

        Epoch is assigned by the last boundary whose step <= current step.
        Val metrics are merged by exact step match.
        """
        # Training DataFrame
        train = pd.DataFrame(self.all_metrics.get("train", {}))
        if train.empty:
            return pd.DataFrame()
        train = train.sort_values("step").reset_index(drop=True)

        # Assign epoch to each train row
        epoch_boundaries = sorted(self.epoch_records, key=lambda x: x[1])
        boundaries = epoch_boundaries.copy()
        current_epoch = 0
        epochs: List[int] = []
        bi = 0
        for s in train["step"]:
            while bi < len(boundaries) and boundaries[bi][1] <= s:
                current_epoch = boundaries[bi][0]
                bi += 1
            epochs.append(current_epoch)
        train.insert(0, "epoch", epochs)

        # Rename train cols
        train_cols = {c: f"train_{c}" for c in train.columns if c not in ("epoch", "step")}
        train = train.rename(columns=train_cols)

        # Validation DataFrame
        val = pd.DataFrame(self.all_metrics.get("val", {}))
        if val.empty:
            return train
        
        val = val.sort_values("step").reset_index(drop=True)
        val_cols = {c: f"val_{c}" for c in val.columns if c != "step"}
        val = val.rename(columns=val_cols)
        return pd.merge(train, val, how="left", on="step")
