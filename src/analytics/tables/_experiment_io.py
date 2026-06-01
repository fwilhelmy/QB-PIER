"""Shared experiment-directory I/O for table extractors (private module)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from omegaconf import OmegaConf

from models.ModelFactory import ModelFactory


def _load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)  # type: ignore[return-value]
    except Exception:
        return {}


def read_hydra_config(model_dir: Path, seed_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load Hydra config from model and/or seed directories."""
    candidates: list[Path] = []
    if seed_dir is not None:
        candidates.append(seed_dir / ".hydra" / "config.yaml")
    candidates.append(model_dir / ".hydra" / "config.yaml")

    merged: Dict[str, Any] = {}
    for path in candidates:
        cfg = _load_yaml_if_exists(path)
        if cfg:
            merged.update(cfg)
            if seed_dir is not None and path.parent.parent == seed_dir:
                return merged
    return merged


def iter_seed_dirs(model_dir: Path) -> Iterator[Path]:
    for item in sorted(model_dir.iterdir()):
        if item.is_dir() and item.name.isdigit():
            yield item


def find_checkpoint(seed_dir: Path, load_mode: str) -> Optional[Path]:
    final_ckpt = seed_dir / "final.ckpt"
    if load_mode == "final":
        return final_ckpt if final_ckpt.exists() else None

    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        return final_ckpt if final_ckpt.exists() else None

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return final_ckpt if final_ckpt.exists() else None

    def loss_score(path: Path) -> float:
        name = path.name
        if "loss=" not in name:
            return float("inf")
        try:
            return float(name.split("loss=")[1].split("-")[0])
        except ValueError:
            return float("inf")

    return sorted(ckpts, key=loss_score)[0]


def load_single_model(
    model_dir: Path,
    seed_dir: Path,
    checkpoint: Path,
    map_location: str = "cpu",
) -> Any:
    model_name = (model_dir / ".model").read_text(encoding="utf-8").strip()
    return ModelFactory.load(
        model_name,
        checkpoint=str(checkpoint),
        map_location=map_location,
        log_dir=str(seed_dir),
    )
