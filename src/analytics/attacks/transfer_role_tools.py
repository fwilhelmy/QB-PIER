"""Notebook-friendly black-box transfer analysis utilities.

The goal of this module is to make the attack matrix, source/target scatter,
and top-k transfer tables use the same configuration system.

Core concepts
-------------
- grain: displayed model level: ``family``, ``architecture`` or ``variant``.
- comparison_mode:
    * ``inter_variant``: exclude exact same source/target variant only.
    * ``inter_architecture``: exclude pairs from the same architecture.
    * ``inter_model`` / ``inter_family``: exclude pairs from the same family.
- variant_aggregate_mode: how variants are collapsed when the displayed grain is
  architecture or family.
- architecture_aggregate_mode: how architectures are collapsed when the displayed
  grain is family.
- split_ansatz_as_architecture: optionally turns e.g. VQM + strong into
  ``VQM-Strong`` and HVQL + random into ``HVQL-Random`` for architecture-grain
  analyses.

This module has no argparse.  Configure normal Python variables in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence
import re

import numpy as np
import pandas as pd

try:  # plotting is optional for table-only tests
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None  # type: ignore[assignment]

from analytics.paths import attack_experiment_dir, find_project_root, output_dir
from analytics.tables.extract_attack import build_pair_metrics
from analytics.tables.model_blacklist import MODEL_BLACKLIST

Grain = Literal["family", "architecture", "variant"]
ComparisonMode = Literal["inter_variant", "inter_architecture", "inter_model", "inter_family"]
AggregateMode = Literal["mean", "median", "best_damage", "best_robust"]
RankMode = Literal["damage", "robust"]

DEFAULT_METRIC = "Transfer ASR"
DEFAULT_TOP_K = 3

METRIC_ALIASES: dict[str, str] = {
    "asr": "Transfer ASR",
    "ASR": "Transfer ASR",
    "transfer_asr": "Transfer ASR",
    "Transfer ASR": "Transfer ASR",
    "robust_accuracy": "Transfer Robust Accuracy",
    "robust acc": "Transfer Robust Accuracy",
    "transfer_robust_accuracy": "Transfer Robust Accuracy",
    "Transfer Robust Accuracy": "Transfer Robust Accuracy",
    "clean_accuracy": "Clean Accuracy",
    "Clean Accuracy": "Clean Accuracy",
    "accuracy_drop": "Accuracy Drop",
    "Accuracy Drop": "Accuracy Drop",
    "flip_rate": "Flip Rate",
    "Flip Rate": "Flip Rate",
    "mean_confidence_drop": "Mean Confidence Drop",
    "Mean Confidence Drop": "Mean Confidence Drop",
    "mean_margin_drop": "Mean Margin Drop",
    "Mean Margin Drop": "Mean Margin Drop",
}

# True means larger value = more damaging. False means smaller value = more damaging.
METRIC_HIGHER_IS_DAMAGE: dict[str, bool] = {
    "Transfer ASR": True,
    "ASR": True,
    "Flip Rate": True,
    "Accuracy Drop": True,
    "Transfer Gap": True,
    "Mean Confidence Drop": True,
    "Mean Margin Drop": True,
    "Transfer Robust Accuracy": False,
    "Robust Accuracy": False,
    "Clean Accuracy": False,
}

SECONDARY_METRICS: tuple[str, ...] = (
    "Transfer ASR",
    "Transfer Robust Accuracy",
    "Clean Accuracy",
    "Accuracy Drop",
    "Flip Rate",
    "Mean Confidence Drop",
    "Mean Margin Drop",
)

ARCH_NORMALIZE: dict[str, str] = {
    "convnet": "CNN",
    "lenet": "CNN",
    "cnn": "CNN",
    "resnet18": "ResNet18",
    "mlpmixer": "MLPMixer",
    "mlp": "MLP",
    "vqm": "VQM",
    "vqm50": "VQM",
    "vqm100": "VQM",
    "vqm200": "VQM",
    "hvqm": "HVQM",
    "hvql": "HVQL",
    "drophvql": "HVQL",
    "reshvql": "HVQL",
    "hpvql": "HPVQL",
    "hpvqm": "HPVQL",
    "hquanv": "HQuanv",
    "hqe": "HQE",
}

FAMILY_COLORS: dict[str, str] = {
    "Classical": "#4C72B0",
    "Hybrid": "#C44E52",
    "Quantum": "#55A868",
    "Unknown": "#888888",
}

FAMILY_MARKERS: dict[str, str] = {
    "Classical": "o",
    "Hybrid": "s",
    "Quantum": "^",
    "Unknown": "o",
}


@dataclass(frozen=True)
class TransferViewConfig:
    """Shared notebook configuration for transfer matrix, scatter, and top-k tables."""

    source_grain: Grain = "architecture"
    target_grain: Grain = "architecture"
    comparison_mode: ComparisonMode = "inter_architecture"
    metric: str = DEFAULT_METRIC
    top_k: int = DEFAULT_TOP_K

    # Hierarchical aggregation.
    # variant_aggregate_mode collapses variant-pair rows into architecture-pair rows.
    # architecture_aggregate_mode collapses architecture-pair rows into family-pair rows.
    variant_aggregate_mode: AggregateMode = "mean"
    architecture_aggregate_mode: AggregateMode = "mean"

    # Treat ansatz choices as architecture labels for quantum/hybrid models.
    split_ansatz_as_architecture: bool = False
    include_no_entangling_as_ansatz: bool = True

    # Blacklist and filters.  blacklist_contains is useful for one-off exclusions.
    model_blacklist: tuple[str, ...] = tuple()
    blacklist_contains: tuple[str, ...] = tuple()
    source_family_filter: str | None = None
    target_family_filter: str | None = None
    source_architecture_filter: str | None = None
    target_architecture_filter: str | None = None
    source_contains: str | None = None
    target_contains: str | None = None

    def with_updates(self, **kwargs: object) -> "TransferViewConfig":
        return replace(self, **kwargs)


# Convenience presets.  Override any field in notebooks with CONFIG.with_updates(...).
VIEW_TEMPLATES: dict[str, TransferViewConfig] = {
    "variant_full": TransferViewConfig(
        source_grain="variant",
        target_grain="variant",
        comparison_mode="inter_variant",
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
    ),
    "architecture_mean": TransferViewConfig(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
    ),
    "architecture_best_damage": TransferViewConfig(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        variant_aggregate_mode="best_damage",
        architecture_aggregate_mode="mean",
    ),
    "architecture_best_robust": TransferViewConfig(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        variant_aggregate_mode="best_robust",
        architecture_aggregate_mode="mean",
    ),
    "architecture_ansatz_mean": TransferViewConfig(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
        split_ansatz_as_architecture=True,
    ),
    "family_mean_architecture_mean_variant": TransferViewConfig(
        source_grain="family",
        target_grain="family",
        comparison_mode="inter_variant",
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
    ),
    "family_mean_architecture_best_damage_variant": TransferViewConfig(
        source_grain="family",
        target_grain="family",
        comparison_mode="inter_variant",
        variant_aggregate_mode="best_damage",
        architecture_aggregate_mode="mean",
    ),
    "family_mean_architecture_best_robust_variant": TransferViewConfig(
        source_grain="family",
        target_grain="family",
        comparison_mode="inter_variant",
        variant_aggregate_mode="best_robust",
        architecture_aggregate_mode="mean",
    ),
    "inter_family_architecture_mean": TransferViewConfig(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_model",
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
    ),
    "top_variants_vs_target_architecture": TransferViewConfig(
        source_grain="variant",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        variant_aggregate_mode="mean",
        architecture_aggregate_mode="mean",
    ),
}


def list_templates() -> pd.DataFrame:
    """Return all predefined templates as a dataframe for notebook display."""
    rows = []
    for name, cfg in VIEW_TEMPLATES.items():
        row = {"template": name}
        row.update(cfg.__dict__)
        rows.append(row)
    return pd.DataFrame(rows)


def get_template(name: str, **overrides: object) -> TransferViewConfig:
    if name not in VIEW_TEMPLATES:
        allowed = ", ".join(sorted(VIEW_TEMPLATES))
        raise KeyError(f"Unknown template {name!r}. Available: {allowed}")
    return VIEW_TEMPLATES[name].with_updates(**overrides)


def resolve_metric(metric: str) -> str:
    key = str(metric).strip()
    return METRIC_ALIASES.get(key, METRIC_ALIASES.get(key.lower(), key))


def metric_higher_is_damage(metric: str) -> bool:
    return METRIC_HIGHER_IS_DAMAGE.get(resolve_metric(metric), True)


def normalize_architecture(arch: object) -> str:
    raw = str(arch) if arch is not None else "Unknown"
    return ARCH_NORMALIZE.get(raw.strip().lower(), raw.strip() or "Unknown")


def parse_canonical_model_name(model_name: object) -> tuple[str, str, str]:
    text = str(model_name).strip().replace("\\", "/")
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 3:
        return parts[0], normalize_architecture(parts[1]), "/".join(parts[2:])
    if len(parts) == 2:
        return parts[0], normalize_architecture(parts[1]), ""
    if len(parts) == 1:
        return "Unknown", normalize_architecture(parts[0]), ""
    return "Unknown", "Unknown", ""


def _clean_family(family: object) -> str:
    raw = str(family).strip()
    lower = raw.lower()
    if lower == "classical":
        return "Classical"
    if lower == "hybrid":
        return "Hybrid"
    if lower == "quantum":
        return "Quantum"
    return raw or "Unknown"


def extract_ansatz_label(config: object, *, include_no_entangling: bool = True) -> str | None:
    """Extract Basic/Random/Strong/NoEntangling from a configuration string."""
    text = str(config).lower().replace(" ", "")
    if not text:
        return None
    if "ansatz=strong" in text or "strong" in text:
        return "Strong"
    if "ansatz=random" in text or "random" in text:
        return "Random"
    if "ansatz=basic" in text or "basic" in text:
        return "Basic"
    if include_no_entangling and ("no_entangling" in text or "noentangling" in text):
        return "NoEntangling"
    return None


def _is_quantum_or_hybrid_family(family: object) -> bool:
    return _clean_family(family) in {"Quantum", "Hybrid"}


def _as_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _mode_or_first(s: pd.Series) -> object:
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iat[0] if not m.empty else s.iat[0]


def _format_value(value: object, metric: str) -> str:
    try:
        v = float(value)
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "--"
    # Most attack metrics are stored as fractions; margin metrics can be > 1.
    if abs(v) <= 1.5:
        return f"{100.0 * v:.1f}%"
    return f"{v:.2f}"


def format_rank_cell(label: object, value: object, metric: str) -> str:
    return f"{label} ({_format_value(value, metric)})"


def attach_model_metadata(pair_df: pd.DataFrame, config: TransferViewConfig | None = None) -> pd.DataFrame:
    """Attach normalized source/target family, architecture, variant, and display labels."""
    cfg = config or TransferViewConfig()
    df = pair_df.copy()
    for side in ("Source", "Target"):
        model_col = f"{side} Model"
        parsed = df[model_col].map(parse_canonical_model_name)
        df[f"{side} Family Norm"] = parsed.map(lambda x: _clean_family(x[0]))
        df[f"{side} Architecture Base"] = parsed.map(lambda x: normalize_architecture(x[1]))
        df[f"{side} Variant Norm"] = parsed.map(lambda x: x[2])

        if f"{side} Family" in df.columns:
            df[f"{side} Family Norm"] = df[f"{side} Family"].fillna(df[f"{side} Family Norm"]).map(_clean_family)
        if f"{side} Architecture" in df.columns:
            df[f"{side} Architecture Base"] = df[f"{side} Architecture"].fillna(df[f"{side} Architecture Base"]).map(normalize_architecture)
        if f"{side} Configuration" in df.columns:
            df[f"{side} Variant Norm"] = df[f"{side} Configuration"].fillna(df[f"{side} Variant Norm"]).astype(str)

        arch = df[f"{side} Architecture Base"].astype(str)
        if cfg.split_ansatz_as_architecture:
            ansatz = df[f"{side} Variant Norm"].map(
                lambda x: extract_ansatz_label(x, include_no_entangling=cfg.include_no_entangling_as_ansatz)
            )
            mask = ansatz.notna() & df[f"{side} Family Norm"].map(_is_quantum_or_hybrid_family)
            arch = arch.where(~mask, arch + "-" + ansatz.fillna("").astype(str))
        df[f"{side} Architecture Norm"] = arch
        df[f"{side} Architecture Label"] = (
            df[f"{side} Family Norm"].astype(str) + "/" + df[f"{side} Architecture Norm"].astype(str)
        )
        df[f"{side} Variant Label"] = df[model_col].astype(str)
    return df


def _black_box_pairs(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "Is Self Attack" in work.columns:
        work = work[~work["Is Self Attack"].astype(bool)]
    elif "WB/BB Flag" in work.columns:
        work = work[work["WB/BB Flag"].astype(str).str.upper().eq("BB")]
    if {"Source Model", "Target Model"}.issubset(work.columns):
        work = work[work["Source Model"].astype(str) != work["Target Model"].astype(str)]
    return work


def _contains_filter(series: pd.Series, term: str | None) -> pd.Series:
    if term is None or not str(term).strip():
        return pd.Series(True, index=series.index)
    return series.astype(str).str.contains(str(term), case=False, regex=False, na=False)


def _apply_blacklist(df: pd.DataFrame, config: TransferViewConfig) -> pd.DataFrame:
    bad_exact = {str(x) for x in MODEL_BLACKLIST} | {str(x) for x in config.model_blacklist}
    work = df.copy()
    if bad_exact:
        work = work[~work["Source Model"].astype(str).isin(bad_exact)]
        work = work[~work["Target Model"].astype(str).isin(bad_exact)]
    contains_terms = [str(x) for x in config.blacklist_contains if str(x).strip()]
    for term in contains_terms:
        mask = work["Source Model"].astype(str).str.contains(term, case=False, regex=False, na=False)
        mask |= work["Target Model"].astype(str).str.contains(term, case=False, regex=False, na=False)
        work = work[~mask]
    return work


def filter_transfer_pairs(pair_df: pd.DataFrame, config: TransferViewConfig) -> pd.DataFrame:
    """Apply black-box, blacklist, comparison-mode, and optional filters."""
    metric = resolve_metric(config.metric)
    if metric not in pair_df.columns:
        raise KeyError(f"Metric {metric!r} not found. Available: {list(pair_df.columns)}")

    df = attach_model_metadata(_black_box_pairs(pair_df), config)
    df = _apply_blacklist(df, config)

    mode = config.comparison_mode
    if mode == "inter_variant":
        pass
    elif mode == "inter_architecture":
        df = df[df["Source Architecture Label"] != df["Target Architecture Label"]]
    elif mode in {"inter_model", "inter_family"}:
        # Notebook convention requested by Felix: inter_model excludes same family.
        df = df[df["Source Family Norm"] != df["Target Family Norm"]]
    else:
        raise ValueError(f"Unknown comparison_mode: {mode!r}")

    if config.source_family_filter:
        df = df[df["Source Family Norm"].str.lower() == _clean_family(config.source_family_filter).lower()]
    if config.target_family_filter:
        df = df[df["Target Family Norm"].str.lower() == _clean_family(config.target_family_filter).lower()]
    if config.source_architecture_filter:
        arch = normalize_architecture(config.source_architecture_filter)
        df = df[df["Source Architecture Norm"].str.lower() == arch.lower()]
    if config.target_architecture_filter:
        arch = normalize_architecture(config.target_architecture_filter)
        df = df[df["Target Architecture Norm"].str.lower() == arch.lower()]

    df = df[_contains_filter(df["Source Model"], config.source_contains)]
    df = df[_contains_filter(df["Target Model"], config.target_contains)]
    df = df.copy()
    for col in set(SECONDARY_METRICS) | {metric}:
        if col in df.columns:
            df[col] = _as_numeric(df, col)
    df["_metric_value"] = _as_numeric(df, metric)
    return df.dropna(subset=["_metric_value"])


def _label_col(side: Literal["Source", "Target"], grain: Grain) -> str:
    if grain == "family":
        return f"{side} Family Norm"
    if grain == "architecture":
        return f"{side} Architecture Label"
    if grain == "variant":
        return f"{side} Variant Label"
    raise ValueError(f"Unknown grain: {grain!r}")


def _metric_aggregate(s: pd.Series, metric: str, mode: AggregateMode) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    if mode == "mean":
        return float(vals.mean())
    if mode == "median":
        return float(vals.median())
    higher_damage = metric_higher_is_damage(metric)
    if mode == "best_damage":
        return float(vals.max() if higher_damage else vals.min())
    if mode == "best_robust":
        return float(vals.min() if higher_damage else vals.max())
    raise ValueError(f"Unknown aggregate mode: {mode!r}")


def _aggregate_stage(
    df: pd.DataFrame,
    keys: Sequence[str],
    *,
    metric: str,
    mode: AggregateMode,
) -> pd.DataFrame:
    if not keys:
        raise ValueError("Aggregation keys cannot be empty")
    metric_cols = [c for c in SECONDARY_METRICS if c in df.columns]
    if metric not in metric_cols and metric in df.columns:
        metric_cols.append(metric)

    rows: list[dict[str, object]] = []
    grouped = df.groupby(list(keys), dropna=False, sort=True)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {k: v for k, v in zip(keys, key)}
        for col in metric_cols:
            # Use the requested mode for the primary metric, mean for secondary context.
            row[col] = _metric_aggregate(sub[col], col, mode if col == metric else "mean")
        row["n_pairs"] = int(len(sub))
        row["n_source_variants"] = int(sub["Source Model"].nunique()) if "Source Model" in sub.columns else np.nan
        row["n_target_variants"] = int(sub["Target Model"].nunique()) if "Target Model" in sub.columns else np.nan
        row["source_family"] = _mode_or_first(sub["Source Family Norm"])
        row["target_family"] = _mode_or_first(sub["Target Family Norm"])
        rows.append(row)
    return pd.DataFrame(rows)


def _stage1_keys(config: TransferViewConfig) -> list[str]:
    keys: list[str] = []
    for side, grain in (("Source", config.source_grain), ("Target", config.target_grain)):
        if grain == "family":
            keys.extend([f"{side} Family Norm", f"{side} Architecture Label"])
        elif grain == "architecture":
            keys.append(f"{side} Architecture Label")
        elif grain == "variant":
            keys.append(f"{side} Variant Label")
        else:
            raise ValueError(f"Unknown grain: {grain!r}")
    return keys


def _final_keys(config: TransferViewConfig) -> tuple[list[str], str, str]:
    source_key = _label_col("Source", config.source_grain)
    target_key = _label_col("Target", config.target_grain)
    return [source_key, target_key], source_key, target_key


def aggregate_transfer_pairs(pair_df: pd.DataFrame, config: TransferViewConfig) -> pd.DataFrame:
    """Aggregate black-box transfer pairs using the shared grain/aggregation config."""
    metric = resolve_metric(config.metric)
    df = filter_transfer_pairs(pair_df, config)
    stage1 = _aggregate_stage(df, _stage1_keys(config), metric=metric, mode=config.variant_aggregate_mode)
    final_keys, source_key, target_key = _final_keys(config)
    if set(final_keys).issubset(stage1.columns) and set(stage1.columns).issuperset(final_keys):
        # If any grain is family, stage1 contains extra architecture columns and must be collapsed.
        needs_stage2 = set(stage1.columns).intersection({"Source Architecture Label", "Target Architecture Label"}) - set(final_keys)
        if needs_stage2:
            stage2 = _aggregate_stage(stage1, final_keys, metric=metric, mode=config.architecture_aggregate_mode)
        else:
            stage2 = stage1.copy()
    else:
        stage2 = stage1.copy()

    out = stage2.rename(columns={source_key: "source_group", target_key: "target_group"})
    if metric not in out.columns and "_metric_value" in out.columns:
        out = out.rename(columns={"_metric_value": metric})
    out["metric"] = metric
    out["comparison_mode"] = config.comparison_mode
    out["source_grain"] = config.source_grain
    out["target_grain"] = config.target_grain
    out["variant_aggregate_mode"] = config.variant_aggregate_mode
    out["architecture_aggregate_mode"] = config.architecture_aggregate_mode
    return out


def _sort_for_rank(df: pd.DataFrame, metric: str, rank_mode: RankMode) -> pd.DataFrame:
    higher_damage = metric_higher_is_damage(metric)
    ascending = not higher_damage if rank_mode == "damage" else higher_damage
    return df.sort_values([metric, "source_group", "target_group"], ascending=[ascending, True, True], kind="mergesort")


def build_top_sources_by_target(
    pair_df: pd.DataFrame,
    config: TransferViewConfig,
    *,
    rank_mode: RankMode = "damage",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-k source groups for each target group.

    rank_mode='damage': most damaging sources per target.
    rank_mode='robust': least damaging sources per target.
    """
    metric = resolve_metric(config.metric)
    grouped = aggregate_transfer_pairs(pair_df, config)
    rows: list[pd.DataFrame] = []
    for target, sub in grouped.groupby("target_group", sort=True):
        ranked = _sort_for_rank(sub, metric, rank_mode).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        rows.append(ranked[ranked["rank"] <= config.top_k])
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    wide_rows: list[dict[str, object]] = []
    for target, sub in long.groupby("target_group", sort=True):
        row: dict[str, object] = {"target": target}
        for k in range(1, config.top_k + 1):
            kk = sub[sub["rank"] == k]
            prefix = "top" if rank_mode == "damage" else "least damaging"
            col = f"{prefix} {k} source ({metric})"
            row[col] = "" if kk.empty else format_rank_cell(kk.iloc[0]["source_group"], kk.iloc[0][metric], metric)
        wide_rows.append(row)
    return pd.DataFrame(wide_rows), long


def build_top_targets_by_source(
    pair_df: pd.DataFrame,
    config: TransferViewConfig,
    *,
    rank_mode: RankMode = "robust",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-k target groups for each source group.

    rank_mode='robust': most robust / least damaged targets per source.
    rank_mode='damage': most damaged targets per source.
    """
    metric = resolve_metric(config.metric)
    grouped = aggregate_transfer_pairs(pair_df, config)
    rows: list[pd.DataFrame] = []
    for source, sub in grouped.groupby("source_group", sort=True):
        ranked = _sort_for_rank(sub, metric, rank_mode).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        rows.append(ranked[ranked["rank"] <= config.top_k])
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    wide_rows: list[dict[str, object]] = []
    for source, sub in long.groupby("source_group", sort=True):
        row: dict[str, object] = {"source": source}
        for k in range(1, config.top_k + 1):
            kk = sub[sub["rank"] == k]
            prefix = "robust" if rank_mode == "robust" else "damaged"
            col = f"top {k} {prefix} target ({metric})"
            row[col] = "" if kk.empty else format_rank_cell(kk.iloc[0]["target_group"], kk.iloc[0][metric], metric)
        wide_rows.append(row)
    return pd.DataFrame(wide_rows), long


def build_attack_matrix(pair_df: pd.DataFrame, config: TransferViewConfig) -> pd.DataFrame:
    """Source-group x target-group matrix for the configured metric."""
    metric = resolve_metric(config.metric)
    grouped = aggregate_transfer_pairs(pair_df, config)
    matrix = grouped.pivot(index="source_group", columns="target_group", values=metric)
    return matrix.sort_index(axis=0).sort_index(axis=1)


def build_source_target_scatter(pair_df: pd.DataFrame, config: TransferViewConfig) -> pd.DataFrame:
    """Build source-vs-target profile for a shared grain.

    For best interpretation, set source_grain == target_grain.  The function can
    still be used with asymmetric grains, but labels may not overlap perfectly.
    """
    metric = resolve_metric(config.metric)
    grouped = aggregate_transfer_pairs(pair_df, config)
    outgoing = grouped.groupby("source_group", dropna=False)[metric].mean().rename("outgoing_score")
    incoming = grouped.groupby("target_group", dropna=False)[metric].mean().rename("incoming_score")
    profile = pd.concat([outgoing, incoming], axis=1).reset_index().rename(columns={"index": "group"})

    source_family = grouped.groupby("source_group")["source_family"].agg(_mode_or_first).rename("source_family")
    target_family = grouped.groupby("target_group")["target_family"].agg(_mode_or_first).rename("target_family")
    family = pd.concat([source_family, target_family], axis=1)
    family["family"] = family["source_family"].combine_first(family["target_family"])
    profile = profile.merge(family[["family"]], left_on="group", right_index=True, how="left")
    profile["family"] = profile["family"].fillna("Unknown")
    profile["metric"] = metric
    profile["comparison_mode"] = config.comparison_mode
    profile["source_grain"] = config.source_grain
    profile["target_grain"] = config.target_grain
    profile["variant_aggregate_mode"] = config.variant_aggregate_mode
    profile["architecture_aggregate_mode"] = config.architecture_aggregate_mode
    return profile.sort_values("group").reset_index(drop=True)


def plot_attack_matrix(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(7.0, 0.48 * n_cols), max(5.2, 0.40 * n_rows))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Metric", rotation=270, labelpad=14)
    fig.tight_layout()
    return fig, ax


def plot_source_target_scatter(
    profile: pd.DataFrame,
    *,
    title: str | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] = (8.2, 6.6),
):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    fig, ax = plt.subplots(figsize=figsize)
    for family, sub in profile.groupby("family", dropna=False):
        color = FAMILY_COLORS.get(str(family), "#888888")
        marker = FAMILY_MARKERS.get(str(family), "o")
        ax.scatter(
            sub["outgoing_score"],
            sub["incoming_score"],
            label=str(family),
            s=85,
            alpha=0.86,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.45,
        )
        if annotate:
            for _, row in sub.iterrows():
                ax.annotate(str(row["group"]), (row["outgoing_score"], row["incoming_score"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    if len(profile):
        ax.axvline(profile["outgoing_score"].mean(), linestyle="--", linewidth=1.0, color="0.4")
        ax.axhline(profile["incoming_score"].mean(), linestyle="--", linewidth=1.0, color="0.4")
    metric = profile["metric"].iloc[0] if "metric" in profile.columns and len(profile) else "metric"
    ax.set_xlabel(f"Outgoing score as source ({metric})")
    ax.set_ylabel(f"Incoming score as target ({metric})")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=220)
    return path


def write_outputs(
    out_dir: Path,
    *,
    prefix: str,
    top_sources_wide: pd.DataFrame | None = None,
    top_sources_long: pd.DataFrame | None = None,
    top_targets_wide: pd.DataFrame | None = None,
    top_targets_long: pd.DataFrame | None = None,
    matrix: pd.DataFrame | None = None,
    scatter_profile: pd.DataFrame | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    objects = {
        "top_sources_wide": top_sources_wide,
        "top_sources_long": top_sources_long,
        "top_targets_wide": top_targets_wide,
        "top_targets_long": top_targets_long,
        "attack_matrix": matrix,
        "source_target_scatter_profile": scatter_profile,
    }
    for name, obj in objects.items():
        if obj is None:
            continue
        path = out_dir / f"{prefix}_{name}.csv"
        obj.to_csv(path, index=True if name == "attack_matrix" else False)
        outputs[name] = path
        if name in {"top_sources_wide", "top_targets_wide", "attack_matrix"}:
            tex_path = out_dir / f"{prefix}_{name}.tex"
            try:
                obj.to_latex(tex_path, index=name != "attack_matrix", escape=True)
                outputs[f"{name}_latex"] = tex_path
            except Exception:
                pass
    return outputs


def load_or_build_pair_metrics(
    *,
    pair_metrics_cache: Path | None = None,
    rebuild: bool = False,
    attack_dir: Path | None = None,
    pair_metrics_output: Path | None = None,
    model_blacklist: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load cached pair metrics or build them from attack experiment tables."""
    if pair_metrics_cache is not None:
        pair_metrics_cache = Path(pair_metrics_cache)
        if pair_metrics_cache.exists() and not rebuild:
            if pair_metrics_cache.suffix.lower() == ".parquet":
                return pd.read_parquet(pair_metrics_cache)
            return pd.read_csv(pair_metrics_cache)
    root = find_project_root()
    atk_dir = attack_dir or attack_experiment_dir(root)
    out_path = pair_metrics_output or output_dir(root, "attack_transfer_role_tables") / "pair_metrics.csv"
    pair_df, _ = build_pair_metrics(
        attack_dir=atk_dir,
        output_path=out_path,
        model_blacklist=set(model_blacklist or MODEL_BLACKLIST),
    )
    return pair_df


def describe_config(config: TransferViewConfig) -> pd.DataFrame:
    return pd.DataFrame([config.__dict__]).T.rename(columns={0: "value"})
