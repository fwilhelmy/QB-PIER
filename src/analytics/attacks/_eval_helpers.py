"""Shared helpers for live-model attack evaluation and plotting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf

from datasets import DatasetFactory

log = logging.getLogger(__name__)


class LimitedDataset(torch.utils.data.Dataset):
    """Small in-memory dataset wrapper for fast attack analytics."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, classes: Iterable[str]):
        super().__init__()
        self.data = data
        self.targets = targets
        self.classes = list(classes)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


def model_device(model: object) -> torch.device:
    """Infer the device where ``model`` parameters live."""
    if hasattr(model, "parameters"):
        try:
            param = next(model.parameters())  # type: ignore[attr-defined]
            return param.device
        except StopIteration:
            pass
    return torch.device("cpu")


def ensure_model_metadata(model_name: str, model: object, default_log_dir: str) -> None:
    if not hasattr(model, "model_name"):
        setattr(model, "model_name", model_name)
    if not hasattr(model, "log_dir"):
        setattr(model, "log_dir", getattr(model, "log_dir", model_name))
    if not hasattr(model, "root_dir"):
        setattr(model, "root_dir", getattr(model, "log_dir", default_log_dir))


def load_limited_dataset_from_seed(
    seed_model: object,
    max_samples: int = 128,
) -> LimitedDataset | None:
    """Recreate a small dataset from the Hydra config stored with a seed run."""
    try:
        log_dir = getattr(seed_model, "log_dir", None)
        if log_dir is None:
            raise RuntimeError("Loaded model does not expose a 'log_dir' attribute")
        seed_dir = Path(log_dir)
        cfg_path = seed_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Hydra config not found at {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            raise KeyError("Dataset section missing from Hydra config")

        dataset_name = str(dataset_cfg.get("dataset", "")).lower()
        if not dataset_name:
            raise ValueError("Dataset name missing from config")

        dataset_root = dataset_cfg.get("root", "./data/processed")
        dm = DatasetFactory.create(dataset_name, root=str(dataset_root), train=False)
        full_dataset = dm.dataset

        classes = getattr(
            full_dataset,
            "classes",
            [str(i) for i in range(dataset_cfg.get("classes", 10))],
        )

        limit = min(max_samples, len(full_dataset))
        data_samples, labels = [], []
        for idx in range(limit):
            sample, label = full_dataset[idx]
            data_samples.append(sample)
            labels.append(int(label))

        data_tensor = torch.stack(tuple(data_samples))
        label_tensor = torch.tensor(labels)
        return LimitedDataset(data_tensor, label_tensor, classes)
    except Exception as exc:
        log.warning("Failed to recreate dataset for analytics: %s", exc)
        return None
