from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from analytics.tables.extract_attack import _normalized_auc
from analytics.tables.model_blacklist import filter_attack_aggregate_df

GroupByLevel = Literal["family", "architecture", "variant", "keyword"]
GroupAgg = Literal["mean", "max", "auc"]
BlockBordersBy = Literal["family"]

# Aggregate metric columns per AGENTS.md §4.3 (attack_experiment tables).
ATTACK_TABLE_METRICS: tuple[str, ...] = (
    "robust_accuracy",
    "flip_rate",
    "ASR",
    "mean_confidence_drop_all",
    "mean_margin_drop_all",
    "mean_confidence_drop_cc",
    "mean_margin_drop_cc",
    "transfer_gap",
)

AttackMetric = Literal[
    "robust_accuracy",
    "flip_rate",
    "ASR",
    "mean_confidence_drop_all",
    "mean_margin_drop_all",
    "mean_confidence_drop_cc",
    "mean_margin_drop_cc",
    "transfer_gap",
]

# Metrics that increase as the attack becomes more effective.
METRICS_HIGHER_IS_WORSE: frozenset[str] = frozenset(
    {
        "flip_rate",
        "ASR",
        "mean_confidence_drop_all",
        "mean_margin_drop_all",
        "mean_confidence_drop_cc",
        "mean_margin_drop_cc",
        "transfer_gap",
    }
)


def _metric_increases_with_attack(metric_col: str) -> bool:
    return metric_col in METRICS_HIGHER_IS_WORSE


def available_attack_metrics(df: pd.DataFrame) -> list[str]:
    """Return attack-table metric columns present in *df*."""
    return [m for m in ATTACK_TABLE_METRICS if m in df.columns]


def validate_attack_metric(df: pd.DataFrame, metric: str) -> str:
    """Ensure *metric* is a known attack-table column and exists in *df*."""
    if metric not in ATTACK_TABLE_METRICS:
        allowed = ", ".join(ATTACK_TABLE_METRICS)
        raise ValueError(f"Unknown attack metric {metric!r}. Choose one of: {allowed}")
    if metric not in df.columns:
        present = ", ".join(available_attack_metrics(df)) or "(none)"
        raise ValueError(
            f"Metric column {metric!r} not in dataframe. Available: {present}"
        )
    return metric


@dataclass
class MatrixAxisGroups:
    """
  Grouping and filtering for one matrix axis (rows or columns).

  Structured levels (``family``, ``architecture``, ``variant``) use the
  canonical model path. ``keyword`` mode assigns each model to the first
  group whose pattern appears as a substring in the full model name (case
  insensitive).

  Examples
  --------
  All Quantum models as columns, bucketed by ansatz::

      MatrixAxisGroups(
          by="keyword",
          keywords=["ansatz=strong", "ansatz=basic", "ansatz=random"],
          filter_family="Quantum",
      )

  Only ``Quantum/VQM`` variants, custom group labels::

      MatrixAxisGroups(
          by="keyword",
          keyword_groups={
              "strong": ["ansatz=strong"],
              "basic": ["ansatz=basic"],
              "random": ["ansatz=random", "ansatz=no_entangling"],
          },
          filter_family="Quantum",
          filter_architecture="VQM",
      )

  By-architecture matrix with an extra row/col per best variant (e.g.
  ``Classical/CNN`` = all variants, ``Classical/CNN*`` = best only). Best
  variants remain in the architecture bucket; ``*`` labels are placed right
  after their architecture group. Diagonal uses white-box pairs only::

      MatrixAxisGroups(
          by="architecture",
          extra_groups={
              "Classical/CNN*": ["conv_channels=[32,64], fc_sizes=[128]"],
              "Classical/MLP*": ["hidden_sizes=[1024,512,64,64]"],
              "Quantum/VQM*": ["ansatz=strong, qdepth=200"],
          },
          diagonal_whitebox_only=True,
      )
  """

    by: GroupByLevel = "family"
    keywords: Sequence[str] | None = None
    keyword_groups: dict[str, Sequence[str]] | None = None
    filter_family: str | None = None
    filter_architecture: str | None = None
    filter_contains: str | None = None
    include_other: bool = False
    extra_groups: dict[str, Sequence[str]] | None = None
    diagonal_whitebox_only: bool = False

    def __post_init__(self) -> None:
        if self.by == "keyword" and not self.keywords and not self.keyword_groups:
            raise ValueError(
                "keyword grouping requires keywords=... or keyword_groups={...}"
            )


def cell_effectiveness_max_steepness(
    subdf: pd.DataFrame,
    metric_col: str = "robust_accuracy",
) -> float:
    """
    Attack effectiveness for a single (source, target) pair:
    maximum drop per epsilon step in *metric_col* (larger => more vulnerable).

    subdf: rows for fixed (source_model, target_model) with ``epsilon`` and *metric_col*.
    """
    if subdf.empty or metric_col not in subdf.columns:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    eps = subdf["epsilon"].to_numpy(float)
    acc = pd.to_numeric(subdf[metric_col], errors="coerce").to_numpy(dtype=float)

    if eps.size < 2:
        return 0.0

    max_slope = 0.0
    for i in range(1, len(eps)):
        de = eps[i] - eps[i - 1]
        if de <= 0:
            continue

        if _metric_increases_with_attack(metric_col):
            da = acc[i] - acc[i - 1]
        else:
            da = acc[i - 1] - acc[i]
        if da > 0:
            max_slope = max(max_slope, da / de)

    return float(max_slope)


def cell_effectiveness_max_rel_drop(
    subdf: pd.DataFrame,
    metric_col: str = "robust_accuracy",
) -> float:
    """
    Max relative drop at max epsilon for *metric_col*:
    (value_at_eps0 - min) / value_at_eps0.
    """
    if subdf.empty or metric_col not in subdf.columns:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    acc = pd.to_numeric(subdf[metric_col], errors="coerce").to_numpy(dtype=float)

    if acc.size == 0:
        return 0.0

    clean_acc = acc[0]
    if clean_acc <= 0:
        return 0.0

    if _metric_increases_with_attack(metric_col):
        peak = float(np.nanmax(acc))
        rel_drop = (peak - clean_acc) / clean_acc
    else:
        trough = float(np.nanmin(acc))
        rel_drop = (clean_acc - trough) / clean_acc
    return float(max(rel_drop, 0.0))

def cell_effectiveness_robust_auc(subdf: pd.DataFrame) -> float:
    """Normalized AUC of ``robust_accuracy`` (alias for metric AUC)."""
    return _pair_normalized_auc(subdf, metric_col="robust_accuracy")


def cell_effectiveness_metric_auc(
    subdf: pd.DataFrame,
    metric_col: str = "robust_accuracy",
) -> float:
    """
    Normalized trapezoidal AUC of *metric_col* over epsilon.

    Same definition as attack-table pair compilation (``extract_attack``).
    """
    return _pair_normalized_auc(subdf, metric_col=metric_col)


def make_metric_cell_fn(
    metric: str,
    *,
    style: Literal["auc", "max_steepness", "max_rel_drop"] = "auc",
) -> Callable[[pd.DataFrame], float]:
    """Build a per-pair cell function for a given attack-table *metric*."""
    if style == "auc":

        def _auc(subdf: pd.DataFrame) -> float:
            return cell_effectiveness_metric_auc(subdf, metric_col=metric)

        return _auc
    if style == "max_steepness":

        def _steep(subdf: pd.DataFrame) -> float:
            return cell_effectiveness_max_steepness(subdf, metric_col=metric)

        return _steep
    if style == "max_rel_drop":

        def _rel(subdf: pd.DataFrame) -> float:
            return cell_effectiveness_max_rel_drop(subdf, metric_col=metric)

        return _rel
    raise ValueError(f"Unknown cell style: {style!r}")


def _pair_normalized_auc(
    subdf: pd.DataFrame,
    metric_col: str = "robust_accuracy",
) -> float:
    if subdf.empty or metric_col not in subdf.columns:
        return 0.0
    work = subdf.sort_values("epsilon")
    x = work["epsilon"].to_numpy(dtype=float)
    y = pd.to_numeric(work[metric_col], errors="coerce").to_numpy(dtype=float)
    val = _normalized_auc(x, y)
    return float(val) if np.isfinite(val) else 0.0


def _block_normalized_auc(
    df: pd.DataFrame,
    source_models: list[str],
    target_models: list[str],
    metric_col: str = "robust_accuracy",
    *,
    same_model_pairs_only: bool = False,
) -> float:
    """
    Pool (source, target) pairs in a matrix block: mean metric at each epsilon,
    then normalized AUC (attack-table style).
    """
    frames: list[pd.DataFrame] = []
    for s in source_models:
        for t in target_models:
            if same_model_pairs_only and s != t:
                continue
            sub = df[(df["source_model"] == s) & (df["target_model"] == t)]
            if not sub.empty:
                frames.append(sub)
    if not frames:
        return 0.0

    combined = pd.concat(frames, ignore_index=True)
    pooled = (
        combined.groupby("epsilon", as_index=False)[metric_col]
        .mean(numeric_only=True)
        .sort_values("epsilon")
    )
    x = pooled["epsilon"].to_numpy(dtype=float)
    y = pd.to_numeric(pooled[metric_col], errors="coerce").to_numpy(dtype=float)
    val = _normalized_auc(x, y)
    return float(val) if np.isfinite(val) else 0.0


def cell_effectiveness_auc_drop(subdf: pd.DataFrame) -> float:
    """
    Total robustness loss (AUC-based)
    “How much accuracy do we lose across the whole epsilon range?”
    1 - (AUC(actual accuracy curve) / AUC(ideal flat-at-clean curve)).

    Higher => less robust over the whole epsilon range.
    
    """
    if subdf.empty:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    eps = subdf["epsilon"].to_numpy(float)
    acc = subdf["accuracy"].to_numpy(float)

    if eps.size < 2:
        return 0.0

    clean_acc = acc[0]
    ideal_auc = clean_acc * (eps[-1] - eps[0])
    if ideal_auc <= 0:
        return 0.0

    actual_auc = np.trapz(acc, eps)
    drop = 1.0 - (actual_auc / ideal_auc)
    return float(max(drop, 0.0))


def cell_effectiveness_eps_to_fraction(subdf: pd.DataFrame, frac: float = 0.5) -> float:
    """
    Threshold fragility (epsilon at which it collapses)
    “How quickly can I push you below a given fraction of clean accuracy?”
    
    Invert epsilon where accuracy first <= frac * clean_acc.
    If it never drops below: 0 (robust).
    Higher => breaks earlier => more vulnerable.
    """
    if subdf.empty:
        return 0.0

    subdf = subdf.sort_values("epsilon")
    eps = subdf["epsilon"].to_numpy(float)
    acc = subdf["accuracy"].to_numpy(float)

    if eps.size < 2:
        return 0.0

    clean_acc = acc[0]
    if clean_acc <= 0:
        return 0.0

    thresh = frac * clean_acc
    for e, a in zip(eps[1:], acc[1:]):
        if a <= thresh:
            # bigger score if it fails at small epsilon
            return float(1.0 / max(e, 1e-8))

    return 0.0


def parse_canonical_model_name(model_name: str) -> tuple[str, str, str]:
    """
    Parse canonical model paths: Family/Architecture[/variant knobs].

    Examples
    --------
    Classical/ResNet18 -> (Classical, ResNet18, "")
    Classical/CNN/LeNet -> (Classical, CNN, LeNet)
    Hybrid/HVQL/ansatz=strong, qdepth=50 -> (Hybrid, HVQL, ansatz=strong, qdepth=50)
    """
    if not isinstance(model_name, str) or not model_name.strip():
        return "Unknown", "Unknown", ""

    parts = [p for p in model_name.strip().replace("\\", "/").split("/") if p]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) == 1:
        return "Unknown", parts[0], ""
    return "Unknown", "Unknown", model_name


def model_group_key(model_name: str, group_by: GroupByLevel) -> str:
    """Return the grouping label for *model_name* at the requested level."""
    family, arch, variant = parse_canonical_model_name(model_name)
    if group_by == "family":
        return family
    if group_by == "architecture":
        return f"{family}/{arch}" if family != "Unknown" else arch
    if group_by == "variant":
        if variant:
            return variant
        return f"{family}/{arch}" if family != "Unknown" else arch
    if group_by == "keyword":
        raise ValueError(
            "Use MatrixAxisGroups(by='keyword', keywords=...) for keyword grouping"
        )
    raise ValueError(f"Unknown group_by level: {group_by!r}")


def keyword_groups_ordered(spec: MatrixAxisGroups) -> list[tuple[str, list[str]]]:
    """Return (label, patterns) pairs in display order."""
    if spec.keyword_groups:
        return [(label, list(patterns)) for label, patterns in spec.keyword_groups.items()]
    if spec.keywords:
        return [(kw, [kw]) for kw in spec.keywords]
    return []


def extra_groups_ordered(spec: MatrixAxisGroups) -> list[tuple[str, list[str]]]:
    """Return (label, patterns) for ``extra_groups`` in display order."""
    if not spec.extra_groups:
        return []
    return [(label, list(patterns)) for label, patterns in spec.extra_groups.items()]


def model_matches_axis_filter(model_name: str, spec: MatrixAxisGroups | None) -> bool:
    if spec is None:
        return True
    family, arch, _variant = parse_canonical_model_name(model_name)
    if spec.filter_family and family.lower() != spec.filter_family.strip().lower():
        return False
    if spec.filter_architecture and arch.lower() != spec.filter_architecture.strip().lower():
        return False
    if spec.filter_contains and spec.filter_contains.lower() not in model_name.lower():
        return False
    return True


def _model_matches_patterns(model_name: str, patterns: Sequence[str]) -> bool:
    return any(pattern.lower() in model_name.lower() for pattern in patterns)


def assign_axis_group_key(
    model_name: str, spec: MatrixAxisGroups | None
) -> str | None:
    """
    Primary group label for *model_name*, or None if filtered out / unmatched.

    ``extra_groups`` do not replace this label; they add optional ``*`` rows/cols
    via :func:`models_for_axis_group`.
    """
    if spec is None:
        return model_name
    if not model_matches_axis_filter(model_name, spec):
        return None

    if spec.by != "keyword":
        return model_group_key(model_name, spec.by)

    for label, patterns in keyword_groups_ordered(spec):
        for pattern in patterns:
            if pattern.lower() in model_name.lower():
                return label
    return "_other_" if spec.include_other else None


def family_from_group_label(label: str) -> str:
    """Family segment from a matrix axis label (e.g. ``Classical/CNN`` -> ``Classical``)."""
    return label.split("/")[0] if "/" in label else label


def extra_group_anchor(anchor_label: str, extra_label: str) -> bool:
    """True if *extra_label* should be displayed immediately after *anchor_label*."""
    if extra_label.endswith("*"):
        return extra_label[:-1] == anchor_label
    return False


def family_block_boundaries(axis_labels: Sequence[str]) -> list[float]:
    """Imshow coordinates for lines between contiguous family blocks on one axis."""
    if len(axis_labels) < 2:
        return []
    bounds: list[float] = []
    for i in range(1, len(axis_labels)):
        if family_from_group_label(axis_labels[i]) != family_from_group_label(
            axis_labels[i - 1]
        ):
            bounds.append(i - 0.5)
    return bounds


def _draw_block_borders(
    ax: plt.Axes,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    block_borders_by: BlockBordersBy,
    linewidth: float = 2.5,
    color: str = "white",
) -> None:
    if block_borders_by != "family":
        raise ValueError(f"Unknown block_borders_by: {block_borders_by!r}")
    for x in family_block_boundaries(col_labels):
        ax.axvline(x, color=color, linewidth=linewidth, zorder=10)
    for y in family_block_boundaries(row_labels):
        ax.axhline(y, color=color, linewidth=linewidth, zorder=10)


def models_for_axis_group(
    group_label: str,
    model_names: Sequence[str],
    spec: MatrixAxisGroups,
) -> list[str]:
    """All models belonging to *group_label* on one matrix axis."""
    extra_map = dict(extra_groups_ordered(spec))
    if group_label in extra_map:
        patterns = extra_map[group_label]
        return [
            m
            for m in model_names
            if model_matches_axis_filter(m, spec)
            and _model_matches_patterns(m, patterns)
        ]
    return [
        m
        for m in model_names
        if assign_axis_group_key(m, spec) == group_label
    ]


def ordered_axis_group_labels(
    model_names: Sequence[str],
    spec: MatrixAxisGroups | None,
) -> list[str]:
    """Row/column labels in display order (architecture groups + interleaved extras)."""
    if spec is None:
        return sorted({str(m) for m in model_names})

    if spec.by == "keyword":
        labels = [label for label, _ in keyword_groups_ordered(spec)]
        if spec.include_other:
            labels = [*labels, "_other_"]
        return labels

    base_labels = sorted(
        {
            key
            for m in model_names
            if (key := assign_axis_group_key(m, spec)) is not None
        },
        key=lambda label: (family_from_group_label(label), label),
    )

    if not spec.extra_groups:
        return base_labels

    ordered: list[str] = []
    extra_labels = [label for label, _ in extra_groups_ordered(spec)]
    for base in base_labels:
        ordered.append(base)
        for extra in extra_labels:
            if not extra_group_anchor(base, extra):
                continue
            if models_for_axis_group(extra, model_names, spec):
                ordered.append(extra)

    for extra in extra_labels:
        if extra in ordered:
            continue
        if models_for_axis_group(extra, model_names, spec):
            ordered.append(extra)

    return ordered


def _coerce_axis_spec(
    level: GroupByLevel | None,
    spec: MatrixAxisGroups | None,
) -> MatrixAxisGroups | None:
    if spec is not None:
        return spec
    if level is None:
        return None
    return MatrixAxisGroups(by=level)


def _axis_title_fragment(spec: MatrixAxisGroups | None, axis: str) -> str:
    if spec is None:
        return ""
    parts = [f"{axis}={spec.by}"]
    if spec.filter_family:
        parts.append(f"family={spec.filter_family}")
    if spec.filter_architecture:
        parts.append(f"arch={spec.filter_architecture}")
    if spec.filter_contains:
        parts.append(f"contains={spec.filter_contains!r}")
    if spec.by == "keyword":
        labels = [label for label, _ in keyword_groups_ordered(spec)]
        parts.append("groups=[" + ", ".join(labels) + "]")
    return " ".join(parts)


def _diagonal_whitebox_enabled(
    row_spec: MatrixAxisGroups | None,
    col_spec: MatrixAxisGroups | None,
) -> bool:
    if row_spec is None or col_spec is None:
        return False
    return row_spec.diagonal_whitebox_only and col_spec.diagonal_whitebox_only


def _block_cell_values(
    mat: np.ndarray,
    sources: list[str],
    targets: list[str],
    row_models: list[str],
    col_models: list[str],
    *,
    same_group: bool,
    diagonal_whitebox_only: bool,
) -> list[float]:
    src_index = {name: i for i, name in enumerate(sources)}
    tgt_index = {name: j for j, name in enumerate(targets)}
    vals: list[float] = []
    for s in row_models:
        for t in col_models:
            if same_group and diagonal_whitebox_only and s != t:
                continue
            i = src_index.get(s)
            j = tgt_index.get(t)
            if i is None or j is None:
                continue
            vals.append(float(mat[i, j]))
    return vals


def _aggregate_matrix(
    mat: np.ndarray,
    sources: list[str],
    targets: list[str],
    df: pd.DataFrame,
    *,
    row_spec: MatrixAxisGroups | None,
    col_spec: MatrixAxisGroups | None,
    group_agg: GroupAgg,
    metric_col: str = "robust_accuracy",
) -> tuple[np.ndarray, list[str], list[str]]:
    row_groups = ordered_axis_group_labels(sources, row_spec)
    col_groups = ordered_axis_group_labels(targets, col_spec)
    out = np.zeros((len(row_groups), len(col_groups)), dtype=float)
    diagonal_wb = _diagonal_whitebox_enabled(row_spec, col_spec)

    for i, g_row in enumerate(row_groups):
        row_models = (
            models_for_axis_group(g_row, sources, row_spec)
            if row_spec is not None
            else [g_row]
        )
        for j, g_col in enumerate(col_groups):
            col_models = (
                models_for_axis_group(g_col, targets, col_spec)
                if col_spec is not None
                else [g_col]
            )
            same_group = g_row == g_col
            if group_agg == "auc":
                out[i, j] = _block_normalized_auc(
                    df,
                    row_models,
                    col_models,
                    metric_col=metric_col,
                    same_model_pairs_only=same_group and diagonal_wb,
                )
                continue
            vals = _block_cell_values(
                mat,
                sources,
                targets,
                row_models,
                col_models,
                same_group=same_group,
                diagonal_whitebox_only=diagonal_wb,
            )
            if not vals:
                continue
            if group_agg == "max":
                out[i, j] = float(np.max(vals))
            else:
                out[i, j] = float(np.mean(vals))

    return out, row_groups, col_groups


def plot_model_attack_effectiveness_matrix(
    df: pd.DataFrame,
    out_dir: str,
    cell_fn: Callable[[pd.DataFrame], float] = cell_effectiveness_max_steepness,
    normalize: bool = True,
    cmap: str = "viridis",
    fname: str = "attack_effectiveness_matrix.png",
    annotate: bool = True,
    group_by: GroupByLevel | None = None,
    group_rows_by: GroupByLevel | None = None,
    group_cols_by: GroupByLevel | None = None,
    group_rows: MatrixAxisGroups | None = None,
    group_cols: MatrixAxisGroups | None = None,
    group_agg: GroupAgg = "mean",
    metric: str = "robust_accuracy",
    block_borders_by: BlockBordersBy | None = None,
    block_border_color: str = "white",
    block_border_linewidth: float = 2.5,
) -> None:
    """
    Build and plot a matrix of attack effectiveness between models.

    Rows : adversarial source models (or row groups)
    Cols : evaluated target models (or column groups)
    Cell : cell_fn(df_pair) for that (source, target)

    If normalize=True, divide all positive values by the global max so the
    strongest vulnerability = 1.0.

    Parameters
    ----------
    group_by:
        Shorthand for both axes: family, architecture, variant, or keyword
        (keyword still needs *group_rows* / *group_cols* with patterns).
    group_rows_by, group_cols_by:
        Simple per-axis level (ignored when *group_rows* / *group_cols* set).
    group_rows, group_cols:
        Full axis spec: grouping mode, keyword buckets, and optional filters
        (``filter_family``, ``filter_architecture``, ``filter_contains``).
    group_agg:
        When grouping: ``mean`` / ``max`` over per-model *cell_fn* scores, or
        ``auc`` to pool epsilon curves in each block and apply normalized AUC
        (same as attack-table pair metrics). With ``auc`` and no grouping,
        each cell uses normalized AUC of *metric* vs epsilon.
    metric:
        Attack-table aggregate column (AGENTS.md §4.3): ``robust_accuracy``,
        ``flip_rate``, ``ASR``, ``mean_confidence_drop_all``, etc. Used for
        ``group_agg='auc'``. For ``mean``/``max``, set *cell_fn* via
        ``make_metric_cell_fn(metric, style=...)`` or pass *metric_col* to
        cell helpers.
    block_borders_by:
        If ``"family"``, draw thick lines between contiguous family blocks on
        both axes (e.g. Classical | Hybrid | Quantum).
    block_border_color, block_border_linewidth:
        Style for family block separators.
    """
    os.makedirs(out_dir, exist_ok=True)
    metric = validate_attack_metric(df, metric)
    df = filter_attack_aggregate_df(df)

    row_spec = group_rows if group_rows is not None else _coerce_axis_spec(
        group_rows_by if group_rows_by is not None else group_by, None
    )
    col_spec = group_cols if group_cols is not None else _coerce_axis_spec(
        group_cols_by if group_cols_by is not None else group_by, None
    )

    sources = sorted(df["source_model"].unique())
    targets = sorted(df["target_model"].unique())

    n_src, n_tgt = len(sources), len(targets)
    mat = np.zeros((n_src, n_tgt), dtype=float)
    use_auc_cells = group_agg == "auc" and row_spec is None and col_spec is None

    if not (group_agg == "auc" and (row_spec is not None or col_spec is not None)):
        for i, s in enumerate(sources):
            for j, t in enumerate(targets):
                subdf = df[(df["source_model"] == s) & (df["target_model"] == t)]
                if use_auc_cells:
                    mat[i, j] = _pair_normalized_auc(subdf, metric_col=metric)
                else:
                    mat[i, j] = cell_fn(subdf)

    row_labels = list(sources)
    col_labels = list(targets)
    if row_spec is not None or col_spec is not None:
        mat, row_labels, col_labels = _aggregate_matrix(
            mat,
            sources,
            targets,
            df,
            row_spec=row_spec,
            col_spec=col_spec,
            group_agg=group_agg,
            metric_col=metric,
        )

    n_src, n_tgt = len(row_labels), len(col_labels)

    if normalize:
        max_val = np.max(mat)
        mat_norm = mat / max_val if max_val > 0 else mat
    else:
        mat_norm = mat

    title_suffix = ""
    if row_spec is not None or col_spec is not None:
        parts = []
        row_frag = _axis_title_fragment(row_spec, "rows")
        col_frag = _axis_title_fragment(col_spec, "cols")
        if row_frag:
            parts.append(row_frag)
        if col_frag:
            parts.append(col_frag)
        parts.append(f"{group_agg}[{metric}]")
        title_suffix = f" ({'; '.join(parts)})"
    elif group_agg == "auc" or use_auc_cells:
        title_suffix = f" (auc[{metric}])"

    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * n_tgt, 1.2 + 0.5 * n_src))

    im = ax.imshow(
        mat_norm,
        interpolation="nearest",
        cmap=cmap,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(range(n_tgt))
    ax.set_xticklabels([str(t) for t in col_labels], rotation=45, ha="right")

    ax.set_yticks(range(n_src))
    ax.set_yticklabels([str(s) for s in row_labels])

    ax.set_xlabel("Target model (victim)", fontsize=9, labelpad=8)
    ax.set_ylabel("Source model (attacker)", fontsize=9, labelpad=8)

    ax.set_title(
        f"Model×Model attack effectiveness (normalised){title_suffix}",
        fontsize=10,
    )

    # light grid to make the matrix easier to read
    ax.set_xticks(np.arange(-0.5, n_tgt, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_src, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    if block_borders_by is not None:
        _draw_block_borders(
            ax,
            row_labels,
            col_labels,
            block_borders_by=block_borders_by,
            linewidth=block_border_linewidth,
            color=block_border_color,
        )

    # annotate cells with numeric values (suppress tiny ones)
    if annotate:
        for i in range(n_src):
            for j in range(n_tgt):
                val = mat_norm[i, j]
                if val > 0.03:  # threshold to avoid visual clutter
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if group_agg == "auc" or use_auc_cells:
        cbar_label = f"Normalised AUC ({metric})"
    else:
        cbar_label = "Normalised vulnerability"
    cbar.set_label(cbar_label, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=220)
    plt.close(fig)
