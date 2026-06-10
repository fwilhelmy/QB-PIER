"""Compatibility wrappers for legacy top-source transfer tables.

This module is kept so older imports from ``analytics.attacks.__init__`` do not
break. New notebooks should use ``analytics.attacks.transfer_role_tools``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .transfer_role_tools import (
    TransferViewConfig,
    get_template,
    load_or_build_pair_metrics,
    build_top_sources_by_target,
    write_outputs,
)


@dataclass
class TopSourceOutputs:
    """Legacy container for top-source reports."""

    variant_top_sources: pd.DataFrame | None = None
    architecture_top_sources: pd.DataFrame | None = None
    family_top_sources: pd.DataFrame | None = None
    output_dir: Path | None = None


def build_variant_top_sources(pair_metrics: pd.DataFrame, *, top_k: int = 3, metric: str = "Transfer ASR", **kwargs: Any) -> pd.DataFrame:
    """Legacy helper: target variant -> top source variants."""
    config = TransferViewConfig(
        source_grain="variant",
        target_grain="variant",
        comparison_mode=kwargs.pop("comparison_mode", "inter_variant"),
        metric=metric,
        top_k=top_k,
        **kwargs,
    )
    return build_top_sources_by_target(pair_metrics, config)[1]


def build_architecture_top_sources(pair_metrics: pd.DataFrame, *, top_k: int = 3, metric: str = "Transfer ASR", **kwargs: Any) -> pd.DataFrame:
    """Legacy helper: target architecture -> top source architectures."""
    config = TransferViewConfig(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode=kwargs.pop("comparison_mode", "inter_architecture"),
        metric=metric,
        top_k=top_k,
        **kwargs,
    )
    return build_top_sources_by_target(pair_metrics, config)[1]


def build_family_top_architecture_sources(pair_metrics: pd.DataFrame, *, top_k: int = 3, metric: str = "Transfer ASR", **kwargs: Any) -> pd.DataFrame:
    """Legacy helper: target family -> top source architectures."""
    config = TransferViewConfig(
        source_grain="architecture",
        target_grain="family",
        comparison_mode=kwargs.pop("comparison_mode", "inter_model"),
        metric=metric,
        top_k=top_k,
        **kwargs,
    )
    return build_top_sources_by_target(pair_metrics, config)[1]


def run_top_sources_report(*, pair_metrics: pd.DataFrame | None = None, config: TransferViewConfig | None = None, output_dir: str | Path | None = None, **kwargs: Any) -> TopSourceOutputs:
    """Legacy report runner.

    Prefer using the notebooks and ``transfer_role_tools`` directly. This wrapper
    is intentionally small and exists to preserve imports.
    """
    if config is None:
        config = get_template("architecture_mean", **kwargs)
    if pair_metrics is None:
        pair_metrics = load_or_build_pair_metrics()

    _, variant_wide = build_top_sources_by_target(
        pair_metrics,
        TransferViewConfig(**{**config.__dict__, "source_grain": "variant", "target_grain": "variant"}),
    )
    _, architecture_wide = build_top_sources_by_target(
        pair_metrics,
        TransferViewConfig(**{**config.__dict__, "source_grain": "architecture", "target_grain": "architecture"}),
    )
    _, family_wide = build_top_sources_by_target(
        pair_metrics,
        TransferViewConfig(**{**config.__dict__, "source_grain": "architecture", "target_grain": "family"}),
    )

    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        write_outputs({
            "top_sources_variant_wide": variant_wide,
            "top_sources_architecture_wide": architecture_wide,
            "top_sources_family_wide": family_wide,
        }, out_path)

    return TopSourceOutputs(variant_wide, architecture_wide, family_wide, out_path)
