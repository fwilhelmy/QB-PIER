"""Parameter-count lookup from ``dump_params_size.json`` for scatter point sizing."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PARAMS_DUMP_PATH = Path(__file__).with_name("dump_params_size.json")


@lru_cache(maxsize=1)
def load_params_size_dump(path: str | None = None) -> dict[str, float]:
    """Load canonical model path → parameter count."""
    dump_path = Path(path) if path else PARAMS_DUMP_PATH
    if not dump_path.is_file():
        return {}
    raw = json.loads(dump_path.read_text(encoding="utf-8"))
    return {str(k).strip(): float(v) for k, v in raw.items()}


def _normalize_model_key(name: str) -> str:
    return name.strip().replace("\\", "/")


def lookup_params_total(
    model: str | None,
    dump: dict[str, float],
    *,
    family: str | None = None,
    architecture: str | None = None,
    variant: str | None = None,
) -> float:
    """
    Resolve parameter count for a canonical model path.

    Tries exact keys, ``{family}/{arch}/default``, and median of keys sharing a prefix.
    """
    candidates: list[str] = []

    if isinstance(model, str) and model.strip():
        key = _normalize_model_key(model)
        candidates.append(key)
        parts = [p for p in key.split("/") if p]
        if len(parts) == 2:
            candidates.append(f"{parts[0]}/{parts[1]}/default")

    fam = (family or "").strip()
    arch = (architecture or "").strip()
    var = (variant or "").strip() if isinstance(variant, str) else ""
    if fam and arch:
        if var:
            candidates.append(f"{fam}/{arch}/{var}")
        candidates.append(f"{fam}/{arch}/default")
        candidates.append(f"{fam}/{arch}")

    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in dump:
            return dump[key]

    prefix = candidates[0] if candidates else ""
    if prefix:
        matches = [v for k, v in dump.items() if k == prefix or k.startswith(prefix + "/")]
        if matches:
            return float(np.median(matches))

    if fam and arch and var:
        cfg = {}
        for part in var.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                cfg[k.strip()] = v.strip()
        if cfg.get("ansatz"):
            partial = [
                v
                for k, v in dump.items()
                if k.startswith(f"{fam}/{arch}/") and f"ansatz={cfg['ansatz']}" in k
            ]
            if partial:
                return float(np.median(partial))

    return float("nan")
