"""Bottom-k black-box transfer tables.

This module names the inverse views of the top-k transfer role tables:

- bottom attack sources: least damaging sources for each target.
- bottom robust targets: most damaged / least robust targets for each source.

The implementation delegates ranking and aggregation to ``transfer_role_tools``
so notebooks using top-k and bottom-k stay directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .transfer_role_tools import (
    TransferViewConfig,
    build_top_sources_by_target,
    build_top_targets_by_source,
    get_template,
    load_or_build_pair_metrics,
    resolve_metric,
    write_outputs,
)


@dataclass
class BottomKTransferOutputs:
    """Container for bottom-k transfer role reports."""

    bottom_sources_wide: pd.DataFrame
    bottom_sources_long: pd.DataFrame
    bottom_targets_wide: pd.DataFrame
    bottom_targets_long: pd.DataFrame
    output_paths: dict[str, Path] | None = None


def build_bottom_sources_by_target(
    pair_metrics: pd.DataFrame,
    config: TransferViewConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return least damaging source groups for each target group."""
    return build_top_sources_by_target(pair_metrics, config, rank_mode="robust")


def build_bottom_targets_by_source(
    pair_metrics: pd.DataFrame,
    config: TransferViewConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return most damaged / least robust target groups for each source group."""
    return build_top_targets_by_source(pair_metrics, config, rank_mode="damage")


def run_bottomk_report(
    *,
    pair_metrics: pd.DataFrame | None = None,
    config: TransferViewConfig | None = None,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    **config_overrides: Any,
) -> BottomKTransferOutputs:
    """Build bottom-k source and target tables for a notebook or script."""
    if config is None:
        config = get_template("architecture_mean", **config_overrides)
    elif config_overrides:
        config = config.with_updates(**config_overrides)

    if pair_metrics is None:
        pair_metrics = load_or_build_pair_metrics()

    bottom_sources_wide, bottom_sources_long = build_bottom_sources_by_target(pair_metrics, config)
    bottom_targets_wide, bottom_targets_long = build_bottom_targets_by_source(pair_metrics, config)

    output_paths = None
    if output_dir is not None:
        metric = resolve_metric(config.metric).replace(" ", "_")
        out_prefix = prefix or f"bottomk_{config.source_grain}_to_{config.target_grain}_{config.comparison_mode}_{metric}"
        output_paths = write_outputs(
            Path(output_dir),
            prefix=out_prefix,
            top_sources_wide=bottom_sources_wide,
            top_sources_long=bottom_sources_long,
            top_targets_wide=bottom_targets_wide,
            top_targets_long=bottom_targets_long,
        )

    return BottomKTransferOutputs(
        bottom_sources_wide=bottom_sources_wide,
        bottom_sources_long=bottom_sources_long,
        bottom_targets_wide=bottom_targets_wide,
        bottom_targets_long=bottom_targets_long,
        output_paths=output_paths,
    )
