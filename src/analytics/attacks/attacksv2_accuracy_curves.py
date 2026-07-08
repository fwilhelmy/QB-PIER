"""AttackV2 epsilon-vs-accuracy curve helpers.

These utilities read stored attack aggregate tables and preserve the epsilon
axis, unlike the pair-metric/AUC views in ``attacksv2_transfer_tools``.

No argparse: configure normal Python variables in notebooks.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Literal, Sequence

import numpy as np
import pandas as pd

try:  # plotting is optional for table-only use
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:  # progress bars are optional in lean notebook/test environments
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        return iterable

from analytics.paths import attack_experiment_dir, find_project_root
from analytics.attacks.attacksv2_transfer_tools import AttacksV2Config, ensure_v2
from analytics.attacks.transfer_role_tools import (
    SECONDARY_METRICS,
    attach_model_metadata,
    metric_higher_is_damage,
)
from analytics.tables.model_blacklist import MODEL_BLACKLIST

CurveMode = Literal["source", "target"]

ROBUST_METRIC = "Transfer Robust Accuracy"

__all__ = [
    "CurveMode",
    "ROBUST_METRIC",
    "load_attack_curve_rows",
    "prepare_curve_rows",
    "available_curve_groups",
    "build_accuracy_curve_table",
    "build_custom_accuracy_curve_table",
    "plot_accuracy_curves",
    "safe_path_part",
    "curve_file_stem",
    "curve_image_dir",
    "build_average_overview_curve_table",
    "plot_average_overview_curves",
    "export_all_accuracy_curve_views",
    "make_synthetic_attack_curve_rows",
    "run_smoke_test",
]

_INVALID_PATH_CHARS = r'<>:"/\\|?*'


def load_attack_curve_rows(attack_dir: Path | None = None) -> pd.DataFrame:
    """Load raw per-epsilon aggregate rows from an attack experiment."""
    if attack_dir is None:
        attack_dir = attack_experiment_dir(find_project_root())
    tables_dir = Path(attack_dir) / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"Could not find attack tables directory: {tables_dir}")

    parquet_files = sorted(tables_dir.glob("*.parquet"))
    csv_files = sorted(tables_dir.glob("*.csv"))
    files = parquet_files or csv_files
    if not files:
        raise FileNotFoundError(f"No parquet/csv attack tables found in {tables_dir}")

    frames = [
        pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        for path in files
    ]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("Attack aggregate table is empty.")
    return df


def _curve_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw attack aggregate columns into transfer-tool columns."""
    required = {"seed", "source_model", "target_model", "epsilon", "robust_accuracy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required attack aggregate columns: {missing}")

    out = pd.DataFrame(
        {
            "Seed": df["seed"],
            "Source Model": df["source_model"].astype(str),
            "Target Model": df["target_model"].astype(str),
            "epsilon": pd.to_numeric(df["epsilon"], errors="coerce"),
            ROBUST_METRIC: pd.to_numeric(df["robust_accuracy"], errors="coerce"),
        }
    )
    if "n_samples" in df.columns:
        out["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")
    if "ASR" in df.columns:
        out["Transfer ASR"] = pd.to_numeric(df["ASR"], errors="coerce")
    if "flip_rate" in df.columns:
        out["Flip Rate"] = pd.to_numeric(df["flip_rate"], errors="coerce")
    if "transfer_gap" in df.columns:
        out["Transfer Gap"] = pd.to_numeric(df["transfer_gap"], errors="coerce")
    if "mean_confidence_drop_all" in df.columns:
        out["Mean Confidence Drop"] = pd.to_numeric(df["mean_confidence_drop_all"], errors="coerce")
    if "mean_margin_drop_all" in df.columns:
        out["Mean Margin Drop"] = pd.to_numeric(df["mean_margin_drop_all"], errors="coerce")

    out["Is Self Attack"] = out["Source Model"] == out["Target Model"]
    out["WB/BB Flag"] = np.where(out["Is Self Attack"], "WB", "BB")
    return out.dropna(subset=["epsilon", ROBUST_METRIC])


def _contains_filter(series: pd.Series, term: str | None) -> pd.Series:
    if term is None or not str(term).strip():
        return pd.Series(True, index=series.index)
    return series.astype(str).str.contains(str(term), case=False, regex=False, na=False)


def _apply_blacklist(df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    bad_exact = {str(x) for x in config.model_blacklist}
    if getattr(config, "apply_global_blacklist", True):
        bad_exact |= {str(x) for x in MODEL_BLACKLIST}
    work = df.copy()
    if bad_exact:
        work = work[~work["Source Model"].astype(str).isin(bad_exact)]
        work = work[~work["Target Model"].astype(str).isin(bad_exact)]
    for term in [str(x) for x in config.blacklist_contains if str(x).strip()]:
        mask = work["Source Model"].astype(str).str.contains(term, case=False, regex=False, na=False)
        mask |= work["Target Model"].astype(str).str.contains(term, case=False, regex=False, na=False)
        work = work[~mask]
    return work


def _filter_curve_pairs(
    df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    include_self_attacks: bool,
) -> pd.DataFrame:
    """Apply AttackV2 blacklist, comparison-mode, and optional source/target filters."""
    cfg = ensure_v2(config)
    work = attach_model_metadata(_curve_base_frame(df), cfg)
    if not include_self_attacks:
        work = work[~work["Is Self Attack"].astype(bool)]
        work = work[work["Source Model"].astype(str) != work["Target Model"].astype(str)]
    work = _apply_blacklist(work, cfg)

    mode = cfg.comparison_mode
    if mode == "inter_variant":
        if not include_self_attacks:
            work = work[work["Source Model"].astype(str) != work["Target Model"].astype(str)]
    elif mode == "inter_architecture":
        if include_self_attacks:
            self_rows = work["Source Model"].astype(str) == work["Target Model"].astype(str)
            cross_arch = work["Source Architecture Label"] != work["Target Architecture Label"]
            work = work[self_rows | cross_arch]
        else:
            work = work[work["Source Architecture Label"] != work["Target Architecture Label"]]
    elif mode in {"inter_model", "inter_family"}:
        if include_self_attacks:
            self_rows = work["Source Model"].astype(str) == work["Target Model"].astype(str)
            cross_family = work["Source Family Norm"] != work["Target Family Norm"]
            work = work[self_rows | cross_family]
        else:
            work = work[work["Source Family Norm"] != work["Target Family Norm"]]
    else:
        raise ValueError(f"Unknown comparison_mode: {mode!r}")

    if cfg.source_family_filter:
        work = work[work["Source Family Norm"].str.lower() == str(cfg.source_family_filter).lower()]
    if cfg.target_family_filter:
        work = work[work["Target Family Norm"].str.lower() == str(cfg.target_family_filter).lower()]
    if cfg.source_architecture_filter:
        work = work[work["Source Architecture Norm"].str.lower() == str(cfg.source_architecture_filter).lower()]
    if cfg.target_architecture_filter:
        work = work[work["Target Architecture Norm"].str.lower() == str(cfg.target_architecture_filter).lower()]
    work = work[_contains_filter(work["Source Model"], cfg.source_contains)]
    work = work[_contains_filter(work["Target Model"], cfg.target_contains)]
    return work.dropna(subset=[ROBUST_METRIC, "epsilon"]).copy()


def _filter_to_representatives(df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    if not config.manual_representatives:
        raise ValueError(
            "variant_aggregate_mode='manual_representative' requires "
            "config.manual_representatives=((architecture_substring, variant_substring), ...)"
        )
    keep = pd.Series(True, index=df.index)
    for side in ("Source", "Target"):
        arch = df[f"{side} Architecture Norm"].astype(str)
        variant = df[f"{side} Variant Label"].astype(str)
        for arch_sub, variant_sub in config.manual_representatives:
            in_arch = arch.str.contains(str(arch_sub), case=False, regex=False, na=False)
            is_rep = variant.str.contains(str(variant_sub), case=False, regex=False, na=False)
            keep &= ~(in_arch & ~is_rep)
    return df.loc[keep].copy()


def _metric_aggregate(s: pd.Series, metric: str, mode: str) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    if mode == "none":
        mode = "mean"
    if mode == "mean":
        return float(vals.mean())
    if mode == "median":
        return float(vals.median())
    higher_damage = metric_higher_is_damage(metric)
    if mode == "best_damage":
        return float(vals.max() if higher_damage else vals.min())
    if mode == "best_robust":
        return float(vals.min() if higher_damage else vals.max())
    if mode == "manual_representative":
        return float(vals.mean())
    raise ValueError(f"Unknown aggregate mode: {mode!r}")


def _label_col(side: Literal["Source", "Target"], grain: str) -> str:
    if grain == "family":
        return f"{side} Family Norm"
    if grain == "architecture":
        return f"{side} Architecture Label"
    if grain == "variant":
        return f"{side} Variant Label"
    raise ValueError(f"Unknown grain: {grain!r}")


def _stage1_keys(config: AttacksV2Config) -> list[str]:
    keys: list[str] = ["epsilon"]
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


def _aggregate_stage(
    df: pd.DataFrame,
    keys: Sequence[str],
    *,
    metric: str,
    mode: str,
) -> pd.DataFrame:
    metric_cols = [c for c in SECONDARY_METRICS if c in df.columns]
    if metric not in metric_cols:
        metric_cols.append(metric)
    rows: list[dict[str, object]] = []
    for key, sub in df.groupby(list(keys), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {k: v for k, v in zip(keys, key)}
        for col in metric_cols:
            row[col] = _metric_aggregate(sub[col], col, mode if col == metric else "mean")
        row["mean"] = row[metric]
        row["std"] = float(pd.to_numeric(sub[metric], errors="coerce").std(ddof=0))
        row["n_rows"] = int(len(sub))
        row["n_seeds"] = int(sub["Seed"].nunique()) if "Seed" in sub.columns else np.nan
        row["n_source_variants"] = int(sub["Source Model"].nunique()) if "Source Model" in sub.columns else np.nan
        row["n_target_variants"] = int(sub["Target Model"].nunique()) if "Target Model" in sub.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_curve_rows(df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    cfg = ensure_v2(config)
    if cfg.variant_aggregate_mode == "none" and (
        cfg.source_grain != "variant" or cfg.target_grain != "variant"
    ):
        raise ValueError(
            "variant_aggregate_mode='none' requires source_grain and target_grain to be 'variant'."
        )
    if cfg.variant_aggregate_mode == "manual_representative":
        df = _filter_to_representatives(df, cfg)
        cfg = cfg.with_updates(variant_aggregate_mode="mean")

    stage1 = _aggregate_stage(
        df,
        _stage1_keys(cfg),
        metric=ROBUST_METRIC,
        mode=str(cfg.variant_aggregate_mode),
    )

    source_key = _label_col("Source", cfg.source_grain)
    target_key = _label_col("Target", cfg.target_grain)
    final_keys = ["epsilon", source_key, target_key]
    needs_stage2 = (
        set(stage1.columns).intersection({"Source Architecture Label", "Target Architecture Label"})
        - {source_key, target_key}
    )
    if needs_stage2:
        grouped = _aggregate_stage(
            stage1,
            final_keys,
            metric=ROBUST_METRIC,
            mode=str(cfg.architecture_aggregate_mode),
        )
    else:
        grouped = stage1.copy()

    grouped = grouped.rename(columns={source_key: "source_group", target_key: "target_group"})
    grouped["robust_accuracy"] = grouped[ROBUST_METRIC]
    grouped["metric"] = "robust_accuracy"
    grouped["comparison_mode"] = cfg.comparison_mode
    grouped["source_grain"] = cfg.source_grain
    grouped["target_grain"] = cfg.target_grain
    grouped["variant_aggregate_mode"] = cfg.variant_aggregate_mode
    grouped["architecture_aggregate_mode"] = cfg.architecture_aggregate_mode
    return grouped.sort_values(["epsilon", "source_group", "target_group"]).reset_index(drop=True)


def prepare_curve_rows(
    df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    include_self_attacks: bool = False,
) -> pd.DataFrame:
    """Return all grouped per-epsilon curves after filtering and aggregation."""
    filtered = _filter_curve_pairs(df, ensure_v2(config), include_self_attacks=include_self_attacks)
    if filtered.empty:
        raise ValueError("No rows left after curve filtering.")
    return _aggregate_curve_rows(filtered, ensure_v2(config))


def _selection_mask(series: pd.Series, selected_group: str) -> pd.Series:
    selected = str(selected_group)
    exact = series.astype(str) == selected
    if exact.any():
        return exact
    return series.astype(str).str.contains(selected, case=False, regex=False, na=False)


def _build_accuracy_curve_table_from_grouped(
    grouped: pd.DataFrame,
    *,
    mode: CurveMode,
    selected_group: str,
) -> pd.DataFrame:
    if mode not in {"source", "target"}:
        raise ValueError("mode must be 'source' or 'target'.")
    select_col = "source_group" if mode == "source" else "target_group"
    curve_col = "target_group" if mode == "source" else "source_group"
    out = grouped[_selection_mask(grouped[select_col], selected_group)].copy()
    if out.empty:
        available = sorted(grouped[select_col].astype(str).unique())
        raise ValueError(f"No {select_col} matched {selected_group!r}. Available: {available}")
    out["selected_group"] = out[select_col]
    out["curve_group"] = out[curve_col]
    out["curve_mode"] = mode
    return out.sort_values(["curve_group", "epsilon"]).reset_index(drop=True)


def available_curve_groups(
    df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    include_self_attacks: bool = False,
) -> dict[str, list[str]]:
    """List source and target groups available after filtering."""
    grouped = prepare_curve_rows(df, config, include_self_attacks=include_self_attacks)
    return {
        "source": sorted(grouped["source_group"].astype(str).unique()),
        "target": sorted(grouped["target_group"].astype(str).unique()),
    }


def build_accuracy_curve_table(
    df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    mode: CurveMode,
    selected_group: str,
    include_self_attacks: bool = False,
) -> pd.DataFrame:
    """Build one selected-source or selected-target epsilon accuracy curve table."""
    grouped = prepare_curve_rows(df, config, include_self_attacks=include_self_attacks)
    return _build_accuracy_curve_table_from_grouped(
        grouped,
        mode=mode,
        selected_group=selected_group,
    )


def _comparison_mask(series: pd.Series, comparison_groups: Sequence[str]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for group in comparison_groups:
        mask |= _selection_mask(series, str(group))
    return mask


def build_custom_accuracy_curve_table(
    df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    primary_role: CurveMode,
    primary_group: str,
    comparison_groups: Sequence[str],
    include_self_attacks: bool = False,
) -> pd.DataFrame:
    """Build selected-primary curves against an explicit comparison set.

    ``primary_role='source'`` means ``primary_group`` is the attack source and
    comparison groups are targets.  ``primary_role='target'`` means the primary
    group is the victim/target and comparison groups are sources.
    """
    if primary_role not in {"source", "target"}:
        raise ValueError("primary_role must be 'source' or 'target'.")
    if not comparison_groups:
        raise ValueError("comparison_groups must contain at least one group.")

    curve = build_accuracy_curve_table(
        df,
        config,
        mode=primary_role,
        selected_group=primary_group,
        include_self_attacks=include_self_attacks,
    )
    out = curve[_comparison_mask(curve["curve_group"], comparison_groups)].copy()
    if out.empty:
        available = sorted(curve["curve_group"].astype(str).unique())
        raise ValueError(
            f"No comparison groups matched {list(comparison_groups)!r}. "
            f"Available for {primary_role}={primary_group!r}: {available}"
        )

    order: dict[str, int] = {}
    labels: dict[str, str] = {}
    for idx, requested in enumerate(comparison_groups):
        matches = curve.loc[_selection_mask(curve["curve_group"], str(requested)), "curve_group"]
        for actual in matches.astype(str).unique():
            order.setdefault(actual, idx)
            labels.setdefault(actual, str(requested))
    out["comparison_group"] = out["curve_group"]
    out["comparison_request"] = out["curve_group"].astype(str).map(labels).fillna(out["curve_group"].astype(str))
    out["comparison_order"] = out["curve_group"].astype(str).map(order).fillna(len(order)).astype(int)
    return out.sort_values(["comparison_order", "curve_group", "epsilon"]).reset_index(drop=True)


def plot_accuracy_curves(
    curve_df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    mode: CurveMode,
    selected_group: str,
    title: str | None = None,
    show_std: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Plot robust-accuracy curves for a selected source or target group."""
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    if curve_df.empty:
        raise ValueError("curve_df is empty.")
    cfg = ensure_v2(config)
    if figsize is None:
        n_lines = int(curve_df["curve_group"].nunique())
        figsize = (9.2, max(5.0, min(10.0, 3.8 + 0.18 * n_lines)))

    fig, ax = plt.subplots(figsize=figsize)
    if "comparison_order" in curve_df.columns:
        plot_df = curve_df.sort_values(["comparison_order", "curve_group", "epsilon"])
    else:
        plot_df = curve_df.sort_values(["curve_group", "epsilon"])
    for group, sub in plot_df.groupby("curve_group", sort=False):
        work = sub.sort_values("epsilon")
        x = work["epsilon"].to_numpy(dtype=float)
        y = work["robust_accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=str(group))
        if show_std and "std" in work.columns:
            std = pd.to_numeric(work["std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - std, y + std, alpha=0.12)

    selected_kind = "source" if mode == "source" else "target"
    curve_kind = "targets" if mode == "source" else "sources"
    ax.set_xlabel("FGSM epsilon")
    ax.set_ylabel("Robust accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.set_title(
        title
        or (
            f"Robust accuracy vs epsilon for {selected_kind}: {selected_group}\n"
            f"curves={curve_kind}, {cfg.source_grain}->{cfg.target_grain}, {cfg.comparison_mode}"
        )
    )
    ax.legend(title=curve_kind.title(), frameon=False, fontsize=8, ncols=1)
    fig.tight_layout()
    return fig, ax


def safe_path_part(value: object, *, lowercase: bool = True) -> str:
    """Return a stable filesystem path component for group labels."""
    text = str(value).strip()
    text = "".join("-" if c in _INVALID_PATH_CHARS else c for c in text)
    text = text.replace("=", "-").replace(",", "")
    text = re.sub(r"[\s\[\]\(\)\{\}]+", "_", text)
    text = re.sub(r"-+", "-", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    if lowercase:
        text = text.lower()
    return text or "unknown"


def _group_path_parts(group: object) -> list[str]:
    parts = [p for p in str(group).replace("\\", "/").split("/") if p]
    if not parts:
        return ["unknown"]
    if len(parts) <= 2:
        return [safe_path_part(part) for part in parts]
    return [safe_path_part(parts[0]), safe_path_part(parts[1]), safe_path_part("__".join(parts[2:]))]


def _group_family(group: object) -> str:
    parts = [p for p in str(group).replace("\\", "/").split("/") if p]
    return safe_path_part(parts[0] if parts else "unknown")


def curve_file_stem(group: str, mode: CurveMode) -> str:
    """Filename stem for an individual source/target curve figure."""
    if mode not in {"source", "target"}:
        raise ValueError("mode must be 'source' or 'target'.")
    parts = [p for p in str(group).replace("\\", "/").split("/") if p]
    label_parts = parts[1:] if len(parts) > 1 else parts
    label = "__".join(safe_path_part(part) for part in label_parts) or "unknown"
    role = "as_source" if mode == "source" else "as_target"
    return f"{label}_{role}"


def curve_image_dir(
    out_dir: Path,
    group: str,
    mode: CurveMode,
    *,
    image_subdir: str = "images",
) -> Path:
    """Folder for one group's source/target curve image exports.

    Architecture labels such as ``Classical/CNN`` become
    ``images/classical/as_source`` or ``images/classical/as_target``; the
    model/architecture name is kept in the filename.
    """
    if mode not in {"source", "target"}:
        raise ValueError("mode must be 'source' or 'target'.")
    role_dir = "as_source" if mode == "source" else "as_target"
    return Path(out_dir) / image_subdir / _group_family(group) / role_dir


def _build_average_overview_curve_table_from_grouped(grouped: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = (
        ("source", "source_group", "target_group", "All targets"),
        ("target", "target_group", "source_group", "All sources"),
    )
    for mode, label_col, other_col, other_label in specs:
        overview = (
            grouped.groupby(["epsilon", label_col], dropna=False, sort=True)
            .agg(
                robust_accuracy=("robust_accuracy", "mean"),
                std=("robust_accuracy", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
                n_curves=(other_col, "nunique"),
                n_rows=("n_rows", "sum"),
            )
            .reset_index()
            .rename(columns={label_col: "curve_group"})
        )
        overview["curve_mode"] = mode
        overview["selected_group"] = other_label
        overview["averaged_over"] = other_col
        if mode == "source":
            overview["source_group"] = overview["curve_group"]
            overview["target_group"] = other_label
        else:
            overview["source_group"] = other_label
            overview["target_group"] = overview["curve_group"]
        frames.append(overview)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["curve_mode", "curve_group", "epsilon"]
    ).reset_index(drop=True)


def build_average_overview_curve_table(
    df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    include_self_attacks: bool = False,
) -> pd.DataFrame:
    """Average each source over targets and each target over sources by epsilon."""
    grouped = prepare_curve_rows(df, config, include_self_attacks=include_self_attacks)
    return _build_average_overview_curve_table_from_grouped(grouped)


def _plot_overview_panel(ax, sub: pd.DataFrame, *, title: str, legend_title: str, show_std: bool) -> None:
    for group, part in sub.sort_values(["curve_group", "epsilon"]).groupby("curve_group", sort=True):
        work = part.sort_values("epsilon")
        x = work["epsilon"].to_numpy(dtype=float)
        y = work["robust_accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.7, markersize=3.2, label=str(group))
        if show_std and "std" in work.columns:
            std = pd.to_numeric(work["std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - std, y + std, alpha=0.10)
    ax.set_title(title)
    ax.set_xlabel("FGSM epsilon")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(title=legend_title, frameon=False, fontsize=7, ncols=1)


def plot_average_overview_curves(
    overview_df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    title: str | None = None,
    show_std: bool = False,
    figsize: tuple[float, float] | None = None,
):
    """Plot one two-panel overview for average source and target behavior."""
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    if overview_df.empty:
        raise ValueError("overview_df is empty.")
    cfg = ensure_v2(config)
    if figsize is None:
        max_lines = int(overview_df.groupby("curve_mode")["curve_group"].nunique().max())
        figsize = (15.5, max(5.2, min(11.0, 3.8 + 0.16 * max_lines)))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes_list = list(np.ravel(axes))
    source_df = overview_df[overview_df["curve_mode"] == "source"]
    target_df = overview_df[overview_df["curve_mode"] == "target"]
    _plot_overview_panel(
        axes_list[0],
        source_df,
        title="Average behavior as source",
        legend_title="Sources",
        show_std=show_std,
    )
    axes_list[0].set_ylabel("Robust accuracy")
    _plot_overview_panel(
        axes_list[1],
        target_df,
        title="Average behavior as target",
        legend_title="Targets",
        show_std=show_std,
    )
    fig.suptitle(
        title
        or (
            "Average robust accuracy vs epsilon\n"
            f"{cfg.source_grain}->{cfg.target_grain}, {cfg.comparison_mode}"
        )
    )
    fig.tight_layout()
    return fig, axes


def export_all_accuracy_curve_views(
    df: pd.DataFrame,
    config: AttacksV2Config,
    out_dir: Path,
    *,
    include_self_attacks: bool = False,
    modes: Sequence[CurveMode] = ("source", "target"),
    csv_subdir: str = "csv",
    image_subdir: str = "images",
    image_formats: Sequence[str] = ("png", "pdf"),
    show_std: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export curve data once and one image per available source/target group.

    CSV outputs are intentionally flat under ``out_dir/csv``.  Images are
    grouped as ``out_dir/images/<family>/as_source`` and ``.../as_target``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    cfg = ensure_v2(config)
    out_dir = Path(out_dir)
    csv_dir = out_dir / csv_subdir
    csv_dir.mkdir(parents=True, exist_ok=True)

    grouped = prepare_curve_rows(df, cfg, include_self_attacks=include_self_attacks)
    groups_by_mode = {
        "source": sorted(grouped["source_group"].astype(str).unique()),
        "target": sorted(grouped["target_group"].astype(str).unique()),
    }

    curve_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    selected_modes = [mode for mode in modes if mode in {"source", "target"}]
    for mode in selected_modes:
        for group in tqdm(groups_by_mode[mode], desc=f"Exporting {mode} curves"):
            curve = _build_accuracy_curve_table_from_grouped(grouped, mode=mode, selected_group=group)
            curve_frames.append(curve)

            name = f"{curve_file_stem(group, mode)}__{cfg.slug()}"
            title_role = "source" if mode == "source" else "target"
            comparison_side = "targets" if mode == "source" else "sources"
            fig, _ax = plot_accuracy_curves(
                curve,
                cfg,
                mode=mode,
                selected_group=group,
                title=f"{group} as {title_role}: all {comparison_side}",
                show_std=show_std,
            )

            image_dir = curve_image_dir(out_dir, group, mode, image_subdir=image_subdir)
            image_dir.mkdir(parents=True, exist_ok=True)
            paths: dict[str, Path] = {}
            for ext in image_formats:
                path = image_dir / f"{name}.{ext}"
                fig.savefig(path, bbox_inches="tight", dpi=220)
                paths[str(ext)] = path
            if plt is not None:
                plt.close(fig)

            manifest_rows.append(
                {
                    "mode": mode,
                    "selected_group": group,
                    "n_rows": int(len(curve)),
                    "n_curves": int(curve["curve_group"].nunique()),
                    "file_stem": name,
                    "image_dir": str(image_dir),
                    **{f"{ext}_path": str(path) for ext, path in paths.items()},
                }
            )

    if not curve_frames:
        raise ValueError("No curve views were exported.")

    combined = pd.concat(curve_frames, ignore_index=True)
    combined_path = csv_dir / f"all_accuracy_curves__{cfg.slug()}.csv"
    combined.to_csv(combined_path, index=False)

    overview = _build_average_overview_curve_table_from_grouped(grouped)
    overview_path = csv_dir / f"average_accuracy_overview__{cfg.slug()}.csv"
    overview.to_csv(overview_path, index=False)
    overview_name = f"average_source_target_overview__{cfg.slug()}"
    overview_dir = out_dir / image_subdir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    fig, _axes = plot_average_overview_curves(overview, cfg)
    overview_paths: dict[str, Path] = {}
    for ext in image_formats:
        path = overview_dir / f"{overview_name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        overview_paths[str(ext)] = path
    if plt is not None:
        plt.close(fig)
    manifest_rows.append(
        {
            "mode": "overview",
            "selected_group": "all",
            "n_rows": int(len(overview)),
            "n_curves": int(overview["curve_group"].nunique()),
            "file_stem": overview_name,
            "image_dir": str(overview_dir),
            "overview_csv_path": str(overview_path),
            **{f"{ext}_path": str(path) for ext, path in overview_paths.items()},
        }
    )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = csv_dir / f"accuracy_curve_export_manifest__{cfg.slug()}.csv"
    manifest["data_csv_path"] = str(combined_path)
    manifest["average_overview_csv_path"] = str(overview_path)
    manifest.to_csv(manifest_path, index=False)
    return combined, manifest


def make_synthetic_attack_curve_rows() -> pd.DataFrame:
    """Small raw aggregate table for tests and smoke checks."""
    models = [
        "Classical/CNN/LeNet",
        "Classical/CNN/Wide",
        "Classical/MLP/Base",
        "Hybrid/HVQL/ansatz=strong, qdepth=50",
        "Quantum/VQM/ansatz=basic, qdepth=50",
        "Quantum/VQM/ansatz=no_entangling, qdepth=50",
    ]
    epsilons = [0.0, 0.1, 0.2]
    rows: list[dict[str, object]] = []
    for seed in (0, 42):
        for source in models:
            for target in models:
                src_idx = models.index(source)
                tgt_idx = models.index(target)
                for eps in epsilons:
                    penalty = 0.08 * src_idx + 0.035 * tgt_idx + 0.5 * eps
                    if source == target:
                        penalty *= 0.4
                    rows.append(
                        {
                            "seed": seed,
                            "source_model": source,
                            "target_model": target,
                            "epsilon": eps,
                            "n_samples": 100,
                            "robust_accuracy": max(0.0, 0.98 - penalty + 0.01 * (seed == 42)),
                            "flip_rate": min(1.0, penalty),
                            "ASR": min(1.0, penalty * 1.2),
                            "transfer_gap": 0.05,
                        }
                    )
    return pd.DataFrame(rows)


def run_smoke_test(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Verify source and target curve tables build on synthetic raw rows."""
    raw = make_synthetic_attack_curve_rows()
    cfg = AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        metric=ROBUST_METRIC,
    )
    source_curve = build_accuracy_curve_table(
        raw,
        cfg,
        mode="source",
        selected_group="Classical/CNN",
    )
    target_curve = build_accuracy_curve_table(
        raw,
        cfg,
        mode="target",
        selected_group="Hybrid/HVQL",
    )
    assert set(source_curve["source_group"]) == {"Classical/CNN"}
    assert set(target_curve["target_group"]) == {"Hybrid/HVQL"}
    assert "Classical/CNN" not in set(source_curve["target_group"])
    assert source_curve["epsilon"].nunique() == 3
    if verbose:
        print(f"[smoke] source_curve: {source_curve.shape}")
        print(f"[smoke] target_curve: {target_curve.shape}")
        print("[smoke] all checks passed")
    return {"source_curve": source_curve, "target_curve": target_curve}
