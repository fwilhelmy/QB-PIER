"""Black-box transfer analytics V2.

Shared backend for the four ``attacksv2_*`` notebooks:

- source-target scatter   (How strong is a group as a source and as a target?)
- attack matrix           (Who attacks whom?)
- top-k rankings          (Which patterns keep recurring?)
- directional asymmetry   (Where is transfer non-reciprocal?)

This module extends :mod:`analytics.attacks.transfer_role_tools` rather than
replacing it: the same filtering, blacklist, grain and aggregation backend is
used everywhere, configured through :class:`AttacksV2Config` (a superset of
``TransferViewConfig``).

No argparse: configure plain Python variables in the notebooks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

try:  # plotting is optional for table-only use
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]

from analytics.paths import find_project_root
from analytics.attacks.transfer_role_tools import (
    TransferViewConfig,
    FAMILY_COLORS,
    FAMILY_MARKERS,
    aggregate_transfer_pairs,
    attach_model_metadata,
    filter_transfer_pairs,
    load_or_build_pair_metrics,  # noqa: F401  (re-exported for notebooks)
    metric_higher_is_damage,
    resolve_metric,
    save_figure,
)

ScatterDisplayMode = Literal["architectures_only", "variants_only", "both"]
RankingMode = Literal[
    "damaging_sources_for_targets",
    "robust_targets_for_sources",
    "least_damaging_sources_for_targets",
    "least_robust_targets_for_sources",
]
TopKFigureVariant = Literal[
    "rank1_vs_topk",
    "rank_decomposition",
    "family_composition",
    "family_split_rank1_vs_topk",
]

FAMILIES: tuple[str, ...] = ("Classical", "Hybrid", "Quantum")

#: ranking_mode -> (query side, candidate side, sort ascending given "higher = damage")
_RANKING_SPEC: dict[str, tuple[str, str, str]] = {
    "damaging_sources_for_targets": ("target", "source", "damage"),
    "robust_targets_for_sources": ("source", "target", "robust"),
    "least_damaging_sources_for_targets": ("target", "source", "robust"),
    "least_robust_targets_for_sources": ("source", "target", "damage"),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttacksV2Config(TransferViewConfig):
    """V2 configuration shared by scatter, matrix, top-k and asymmetry views.

    Inherits every ``TransferViewConfig`` field (source_grain, target_grain,
    comparison_mode, metric, top_k, variant_aggregate_mode,
    architecture_aggregate_mode, split_ansatz_as_architecture,
    model_blacklist, blacklist_contains, ...).

    Additional fields
    -----------------
    blacklist_exact:
        Exact model/variant labels to exclude (merged with ``model_blacklist``).
    split_quantum_ansatz_as_architecture:
        V2 alias for ``split_ansatz_as_architecture``; if set, it wins.
    selected_rows / selected_columns:
        Optional explicit row/column (source/target group) selection and order.
    scatter_display_mode:
        ``architectures_only`` | ``variants_only`` | ``both``.
    manual_representatives:
        For ``variant_aggregate_mode='manual_representative'``: tuple of
        ``(architecture_substring, variant_substring)`` rules.  For each rule,
        only variants matching ``variant_substring`` are kept for that
        architecture; architectures without a rule keep all variants.
    output_dir:
        Base export directory; defaults to ``outputs/attacksv2_transfer/``.
    """

    blacklist_exact: tuple[str, ...] = tuple()
    split_quantum_ansatz_as_architecture: bool | None = None
    selected_rows: tuple[str, ...] | None = None
    selected_columns: tuple[str, ...] | None = None
    scatter_display_mode: ScatterDisplayMode = "architectures_only"
    manual_representatives: tuple[tuple[str, str], ...] = tuple()
    output_dir: str | None = None

    def __post_init__(self) -> None:
        # Merge the V2 exact blacklist into the shared one used by the backend.
        if self.blacklist_exact:
            merged = tuple(sorted({*map(str, self.model_blacklist), *map(str, self.blacklist_exact)}))
            object.__setattr__(self, "model_blacklist", merged)
        # Keep the two split flags consistent; the V2 name wins when set.
        if self.split_quantum_ansatz_as_architecture is None:
            object.__setattr__(
                self, "split_quantum_ansatz_as_architecture", bool(self.split_ansatz_as_architecture)
            )
        else:
            object.__setattr__(
                self, "split_ansatz_as_architecture", bool(self.split_quantum_ansatz_as_architecture)
            )

    def with_updates(self, **kwargs: object) -> "AttacksV2Config":
        # Keep the legacy and V2 split-flag names interchangeable in updates.
        if "split_ansatz_as_architecture" in kwargs and "split_quantum_ansatz_as_architecture" not in kwargs:
            kwargs["split_quantum_ansatz_as_architecture"] = kwargs["split_ansatz_as_architecture"]
        return replace(self, **kwargs)

    def slug(self) -> str:
        """Filename-friendly description of the main config axes."""
        metric = resolve_metric(self.metric).replace(" ", "_").lower()
        parts = [
            f"{self.source_grain}_to_{self.target_grain}",
            self.comparison_mode,
            metric,
            f"vagg-{self.variant_aggregate_mode}",
        ]
        if self.split_quantum_ansatz_as_architecture:
            parts.append("ansatz_split")
        return "__".join(parts)


def ensure_v2(config: TransferViewConfig | AttacksV2Config) -> AttacksV2Config:
    """Upgrade a plain TransferViewConfig to AttacksV2Config if needed."""
    if isinstance(config, AttacksV2Config):
        return config
    return AttacksV2Config(**config.__dict__)


# Templates for the standard thesis views.  Override fields with
# ``get_v2_template(name, metric=..., top_k=...)`` or ``cfg.with_updates(...)``.
V2_TEMPLATES: dict[str, AttacksV2Config] = {
    "architecture_main": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        metric="Transfer ASR",
        #blacklist_contains=("HQuanv",),
    ),
    "architecture_with_variants": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        metric="Transfer ASR",
        scatter_display_mode="both",
    ),
    "variant_spread": AttacksV2Config(
        source_grain="variant",
        target_grain="variant",
        comparison_mode="inter_architecture",
        metric="Transfer ASR",
        scatter_display_mode="variants_only",
    ),
    "family_synthesis": AttacksV2Config(
        source_grain="family",
        target_grain="family",
        comparison_mode="inter_family",
        metric="Transfer ASR",
    ),
    "ansatz_split": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        metric="Transfer ASR",
        split_quantum_ansatz_as_architecture=True,
    ),
    "topk_main_architecture": AttacksV2Config(
        source_grain="architecture",
        target_grain="architecture",
        comparison_mode="inter_architecture",
        metric="Transfer ASR",
        top_k=3,
        #blacklist_contains=("HQuanv",),
    ),
}


def get_v2_template(name: str, **overrides: object) -> AttacksV2Config:
    if name not in V2_TEMPLATES:
        allowed = ", ".join(sorted(V2_TEMPLATES))
        raise KeyError(f"Unknown V2 template {name!r}. Available: {allowed}")
    return V2_TEMPLATES[name].with_updates(**overrides)


def list_v2_templates() -> pd.DataFrame:
    rows = []
    for name, cfg in V2_TEMPLATES.items():
        row = {"template": name}
        row.update(cfg.__dict__)
        rows.append(row)
    return pd.DataFrame(rows)


def v2_output_dir(kind: str, config: AttacksV2Config | None = None, root: Path | None = None) -> Path:
    """Resolve ``outputs/attacksv2_transfer/<kind>/`` (or ``config.output_dir/<kind>``)."""
    if config is not None and config.output_dir:
        base = Path(config.output_dir)
    else:
        base = (root or find_project_root()) / "outputs" / "attacksv2_transfer"
    path = base / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Metric direction helpers
# ---------------------------------------------------------------------------

def metric_direction_notes(metric: str) -> dict[str, str]:
    """Human-readable direction semantics for axis labels and captions."""
    metric = resolve_metric(metric)
    if metric_higher_is_damage(metric):
        return {
            "source": f"higher {metric} = more damaging source",
            "target": f"higher {metric} = more vulnerable target",
            "robust_target": f"lower {metric} = more robust target",
        }
    return {
        "source": f"lower {metric} = more damaging source",
        "target": f"lower {metric} = more vulnerable target",
        "robust_target": f"higher {metric} = more robust target",
    }


def _rank_ascending(metric: str, rank_kind: str) -> bool:
    """Sort direction so that rank 1 is 'most damaging' or 'most robust'."""
    higher_damage = metric_higher_is_damage(metric)
    if rank_kind == "damage":
        return not higher_damage
    if rank_kind == "robust":
        return higher_damage
    raise ValueError(f"Unknown rank kind: {rank_kind!r}")


# ---------------------------------------------------------------------------
# Aggregation backend (single entry point for all four views)
# ---------------------------------------------------------------------------

def _filter_to_representatives(pair_df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    """Keep only manually chosen representative variants per architecture."""
    if not config.manual_representatives:
        raise ValueError(
            "variant_aggregate_mode='manual_representative' requires "
            "config.manual_representatives=((architecture_substring, variant_substring), ...)"
        )
    meta = attach_model_metadata(pair_df, config)
    keep = pd.Series(True, index=meta.index)
    for side in ("Source", "Target"):
        arch = meta[f"{side} Architecture Norm"].astype(str)
        variant = meta[f"{side} Variant Label"].astype(str)
        for arch_sub, variant_sub in config.manual_representatives:
            in_arch = arch.str.contains(str(arch_sub), case=False, regex=False, na=False)
            is_rep = variant.str.contains(str(variant_sub), case=False, regex=False, na=False)
            keep &= ~(in_arch & ~is_rep)
    return pair_df.loc[keep[keep].index]


def aggregate_pairs_v2(pair_df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    """Aggregate transfer pairs with V2 aggregation modes.

    Adds ``none`` (variant grain only) and ``manual_representative`` on top of
    the shared mean/median/best_damage/best_robust backend.
    """
    cfg = ensure_v2(config)
    mode = str(cfg.variant_aggregate_mode)
    work = pair_df
    if mode == "none":
        if cfg.source_grain != "variant" or cfg.target_grain != "variant":
            raise ValueError(
                "variant_aggregate_mode='none' requires source_grain and target_grain "
                "to be 'variant' (no variant collapsing happens at variant grain)."
            )
        cfg = cfg.with_updates(variant_aggregate_mode="mean")  # identity at variant grain
    elif mode == "manual_representative":
        work = _filter_to_representatives(pair_df, cfg)
        cfg = cfg.with_updates(variant_aggregate_mode="mean")
    return aggregate_transfer_pairs(work, cfg)


def _group_info(pair_df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    """family/architecture lookup per displayed group label (both sides pooled)."""
    cfg = ensure_v2(config)
    df = filter_transfer_pairs(pair_df, cfg)
    frames = []
    for side, grain in (("Source", cfg.source_grain), ("Target", cfg.target_grain)):
        if grain == "family":
            label = df[f"{side} Family Norm"]
            arch = df[f"{side} Family Norm"]
        elif grain == "architecture":
            label = df[f"{side} Architecture Label"]
            arch = df[f"{side} Architecture Label"]
        else:
            label = df[f"{side} Variant Label"]
            arch = df[f"{side} Architecture Label"]
        frames.append(pd.DataFrame({
            "group": label.astype(str),
            "family": df[f"{side} Family Norm"].astype(str),
            "architecture": arch.astype(str),
        }))
    pooled = pd.concat(frames, ignore_index=True)
    info = (
        pooled.groupby("group", sort=True)
        .agg(family=("family", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
             architecture=("architecture", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"))
        .reset_index()
    )
    return info


def _apply_selection(labels: Sequence[str], selection: Sequence[str] | None) -> list[str]:
    """Filter and order labels by an explicit selection (substring-tolerant)."""
    labels = [str(x) for x in labels]
    if not selection:
        return labels
    ordered: list[str] = []
    for want in selection:
        want_s = str(want)
        exact = [l for l in labels if l == want_s]
        partial = [l for l in labels if want_s.lower() in l.lower()]
        for l in exact or partial:
            if l not in ordered:
                ordered.append(l)
    return ordered


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_table(df: pd.DataFrame, out_dir: Path, name: str, *, index: bool = False,
                 latex: bool = False) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=index)
    paths["csv"] = csv_path
    if latex:
        tex_path = out_dir / f"{name}.tex"
        try:
            df.to_latex(tex_path, index=index, escape=True, float_format="%.3f")
            paths["tex"] = tex_path
        except Exception:
            pass
    return paths


def export_figure(fig, out_dir: Path, name: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        paths[ext] = path
    return paths


# ---------------------------------------------------------------------------
# 1. Source-target scatter
# ---------------------------------------------------------------------------

_FINER_GRAIN = {"family": "architecture", "architecture": "variant", "variant": "variant"}


def _scatter_profile_at(pair_df: pd.DataFrame, cfg: AttacksV2Config, level: str) -> pd.DataFrame:
    metric = resolve_metric(cfg.metric)
    grouped = aggregate_pairs_v2(pair_df, cfg)
    outgoing = grouped.groupby("source_group", dropna=False)[metric].mean().rename("outgoing_score")
    incoming = grouped.groupby("target_group", dropna=False)[metric].mean().rename("incoming_score")
    profile = pd.concat([outgoing, incoming], axis=1)
    profile.index.name = "group"
    profile = profile.reset_index()
    info = _group_info(pair_df, cfg)
    profile = profile.merge(info, on="group", how="left")
    profile["family"] = profile["family"].fillna("Unknown")
    profile["level"] = level
    return profile


def build_source_target_scatter(
    pair_df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    display_mode: ScatterDisplayMode | None = None,
) -> pd.DataFrame:
    """Source potency vs target vulnerability profile.

    Columns: group, level (coarse|fine), parent_group, family, architecture,
    outgoing_score (mean over targets attacked), incoming_score (mean over
    sources attacking it), plus config metadata.
    """
    cfg = ensure_v2(config)
    mode = display_mode or cfg.scatter_display_mode
    metric = resolve_metric(cfg.metric)

    frames: list[pd.DataFrame] = []
    if mode in ("architectures_only", "both"):
        frames.append(_scatter_profile_at(pair_df, cfg, "coarse"))
    if mode in ("variants_only", "both"):
        fine_src = _FINER_GRAIN[cfg.source_grain]
        fine_tgt = _FINER_GRAIN[cfg.target_grain]
        fine_cfg = cfg.with_updates(source_grain=fine_src, target_grain=fine_tgt)
        if str(cfg.variant_aggregate_mode) == "none" and fine_src != "variant":
            fine_cfg = fine_cfg.with_updates(variant_aggregate_mode="mean")
        fine = _scatter_profile_at(pair_df, fine_cfg, "fine" if mode == "both" else "coarse")
        if mode == "both":
            # Drop fine rows identical to coarse rows (variant grain has no finer level).
            coarse_groups = set(frames[0]["group"]) if frames else set()
            fine = fine[~fine["group"].isin(coarse_groups)]
        frames.append(fine)

    profile = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if profile.empty:
        raise ValueError("Scatter profile is empty; check filters/blacklist.")

    # Parent group for fine points (their architecture, or family at family grain).
    profile["parent_group"] = np.where(
        profile["level"] == "fine", profile["architecture"], profile["group"]
    )
    if cfg.source_grain == "family":
        profile.loc[profile["level"] == "fine", "parent_group"] = profile.loc[
            profile["level"] == "fine", "family"
        ]

    profile = profile[_apply_selection_mask(profile["group"], cfg.selected_rows)] if cfg.selected_rows else profile
    profile["metric"] = metric
    profile["comparison_mode"] = cfg.comparison_mode
    profile["source_grain"] = cfg.source_grain
    profile["target_grain"] = cfg.target_grain
    profile["variant_aggregate_mode"] = cfg.variant_aggregate_mode
    profile["display_mode"] = mode
    return profile.sort_values(["level", "group"]).reset_index(drop=True)


def _apply_selection_mask(series: pd.Series, selection: Sequence[str] | None) -> pd.Series:
    if not selection:
        return pd.Series(True, index=series.index)
    mask = pd.Series(False, index=series.index)
    for want in selection:
        mask |= series.astype(str).str.contains(str(want), case=False, regex=False, na=False)
    return mask


def plot_source_target_scatter(
    profile: pd.DataFrame,
    config: AttacksV2Config | None = None,
    *,
    title: str | None = None,
    annotate_coarse: bool = True,
    annotate_fine: bool = False,
    figsize: tuple[float, float] = (8.6, 6.8),
):
    """Scatter of outgoing (x, source potency) vs incoming (y, target exposure).

    In 'both' mode, coarse points are large and labelled; fine points are small
    and semi-transparent behind them.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    metric = str(profile["metric"].iloc[0]) if len(profile) else "metric"
    notes = metric_direction_notes(metric)
    fig, ax = plt.subplots(figsize=figsize)

    fine = profile[profile["level"] == "fine"]
    coarse = profile[profile["level"] == "coarse"]

    for family, sub in fine.groupby("family", dropna=False):
        ax.scatter(
            sub["outgoing_score"], sub["incoming_score"],
            s=26, alpha=0.35, color=FAMILY_COLORS.get(str(family), "#888888"),
            marker=FAMILY_MARKERS.get(str(family), "o"), linewidth=0, zorder=2,
        )
        if annotate_fine:
            for _, row in sub.iterrows():
                ax.annotate(str(row["group"]), (row["outgoing_score"], row["incoming_score"]),
                            xytext=(3, 3), textcoords="offset points", fontsize=6, alpha=0.6)

    for family, sub in coarse.groupby("family", dropna=False):
        ax.scatter(
            sub["outgoing_score"], sub["incoming_score"],
            label=str(family), s=130, alpha=0.92,
            color=FAMILY_COLORS.get(str(family), "#888888"),
            marker=FAMILY_MARKERS.get(str(family), "o"),
            edgecolor="black", linewidth=0.6, zorder=3,
        )
        if annotate_coarse:
            for _, row in sub.iterrows():
                ax.annotate(str(row["group"]), (row["outgoing_score"], row["incoming_score"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8.5,
                            fontweight="bold", zorder=4)

    ref = coarse if len(coarse) else profile
    ax.axvline(ref["outgoing_score"].mean(), linestyle="--", linewidth=1.0, color="0.45", zorder=1)
    ax.axhline(ref["incoming_score"].mean(), linestyle="--", linewidth=1.0, color="0.45", zorder=1)

    ax.set_xlabel(f"Source potency: mean outgoing {metric}\n({notes['source']})")
    ax.set_ylabel(f"Target exposure: mean incoming {metric}\n({notes['target']})")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, title="Family")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Attack matrix
# ---------------------------------------------------------------------------

def build_attack_matrix(pair_df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    """source_group x target_group matrix of the configured metric."""
    cfg = ensure_v2(config)
    metric = resolve_metric(cfg.metric)
    grouped = aggregate_pairs_v2(pair_df, cfg)
    matrix = grouped.pivot(index="source_group", columns="target_group", values=metric)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    rows = _apply_selection(matrix.index, cfg.selected_rows)
    cols = _apply_selection(matrix.columns, cfg.selected_columns)
    return matrix.loc[rows, cols]


def plot_attack_matrix(
    matrix: pd.DataFrame,
    config: AttacksV2Config | None = None,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = "{:.2f}",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(7.0, 0.55 * n_cols + 2.5), max(5.0, 0.45 * n_rows + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    if annotate and n_rows * n_cols <= 600:
        lo = np.nanmin(data) if np.isfinite(data).any() else 0.0
        hi = np.nanmax(data) if np.isfinite(data).any() else 1.0
        mid = lo + 0.55 * (hi - lo)
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if v < mid else "black")
    if title:
        ax.set_title(title)
    metric_label = config.metric if config is not None else "Metric"
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(str(metric_label), rotation=270, labelpad=14)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Top-k rankings
# ---------------------------------------------------------------------------

def build_topk_instances(
    pair_df: pd.DataFrame,
    config: AttacksV2Config,
    *,
    ranking_mode: RankingMode = "damaging_sources_for_targets",
) -> pd.DataFrame:
    """Long-form top-k instances for one ranking mode.

    One row per (query_group, rank): the candidate group ranked at that
    position for that query, with metric value and config metadata.
    """
    cfg = ensure_v2(config)
    if ranking_mode not in _RANKING_SPEC:
        allowed = ", ".join(sorted(_RANKING_SPEC))
        raise ValueError(f"Unknown ranking_mode {ranking_mode!r}. Available: {allowed}")
    query_side, candidate_side, rank_kind = _RANKING_SPEC[ranking_mode]
    metric = resolve_metric(cfg.metric)
    grouped = aggregate_pairs_v2(pair_df, cfg)
    info = _group_info(pair_df, cfg).set_index("group")

    ascending = _rank_ascending(metric, rank_kind)
    query_col = f"{query_side}_group"
    candidate_col = f"{candidate_side}_group"

    rows: list[pd.DataFrame] = []
    for query, sub in grouped.groupby(query_col, sort=True):
        ranked = sub.sort_values([metric, candidate_col], ascending=[ascending, True],
                                 kind="mergesort").copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        rows.append(ranked[ranked["rank"] <= cfg.top_k])
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)

    def _lookup(groups: pd.Series, col: str) -> pd.Series:
        return groups.map(lambda g: info[col].get(str(g), "Unknown"))

    out = pd.DataFrame({
        "query_group": long[query_col].astype(str),
        "candidate_group": long[candidate_col].astype(str),
        "query_family": _lookup(long[query_col], "family"),
        "candidate_family": _lookup(long[candidate_col], "family"),
        "query_architecture": _lookup(long[query_col], "architecture"),
        "candidate_architecture": _lookup(long[candidate_col], "architecture"),
        "rank": long["rank"].astype(int),
        "metric_value": long[metric].astype(float),
        "metric_name": metric,
        "ranking_mode": ranking_mode,
        "comparison_mode": cfg.comparison_mode,
        "source_grain": cfg.source_grain,
        "target_grain": cfg.target_grain,
    })
    if cfg.selected_rows:
        out = out[_apply_selection_mask(out["query_group"], cfg.selected_rows)]
    return out.sort_values(["query_group", "rank"]).reset_index(drop=True)


def summarize_recurring(long_df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    """Recurring-count summary over the candidate side of a top-k long table.

    One row per candidate entity with rank counts, total top-k count, and the
    family composition of the queries it appears against.
    """
    cfg = ensure_v2(config)
    if long_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for entity, sub in long_df.groupby("candidate_group", sort=True):
        row: dict[str, object] = {
            "entity": entity,
            "family": sub["candidate_family"].mode().iat[0],
            "architecture": sub["candidate_architecture"].mode().iat[0],
        }
        for k in range(1, cfg.top_k + 1):
            row[f"rank{k}_count"] = int((sub["rank"] == k).sum())
        row["topk_count"] = int(len(sub))
        rank1 = sub[sub["rank"] == 1]
        for fam in FAMILIES:
            row[f"topk_family_{fam.lower()}_count"] = int((sub["query_family"] == fam).sum())
            row[f"rank1_family_{fam.lower()}_count"] = int((rank1["query_family"] == fam).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(["topk_count", "rank1_count", "entity"],
                               ascending=[False, False, True]).reset_index(drop=True)


def build_recurring_damaging_sources(
    pair_df: pd.DataFrame, config: AttacksV2Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, long) of sources that keep showing up as top-k most damaging."""
    long = build_topk_instances(pair_df, config, ranking_mode="damaging_sources_for_targets")
    return summarize_recurring(long, config), long


def build_recurring_robust_targets(
    pair_df: pd.DataFrame, config: AttacksV2Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, long) of targets that keep showing up as top-k most robust."""
    long = build_topk_instances(pair_df, config, ranking_mode="robust_targets_for_sources")
    return summarize_recurring(long, config), long


def _select_summary_rows(
    summary: pd.DataFrame,
    *,
    selected_entities: Sequence[str] | None = None,
    sort_by: str | Sequence[str] = "topk_count",
    max_rows: int | None = None,
) -> pd.DataFrame:
    df = summary.copy()
    if selected_entities:
        df = df[_apply_selection_mask(df["entity"], selected_entities)]
    if isinstance(sort_by, str):
        if sort_by not in df.columns:
            raise KeyError(f"sort_by column {sort_by!r} not in summary columns")
        df = df.sort_values([sort_by, "entity"], ascending=[False, True])
    else:  # custom explicit order
        order = _apply_selection(df["entity"], list(sort_by))
        df = df.set_index("entity").loc[order].reset_index()
    if max_rows is not None:
        df = df.head(max_rows)
    return df.reset_index(drop=True)


def plot_topk_ranking(
    summary: pd.DataFrame,
    config: AttacksV2Config | None = None,
    *,
    variant: TopKFigureVariant = "rank1_vs_topk",
    role: str = "source",
    title: str | None = None,
    selected_entities: Sequence[str] | None = None,
    sort_by: str | Sequence[str] = "topk_count",
    max_rows: int | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Bar-chart variants of a recurring-count summary.

    variant:
      - rank1_vs_topk: bar = top-k count, darker segment = rank-1 count.
      - rank_decomposition: stacked rank-1 / rank-2 / rank-3 counts.
      - family_composition: stacked by family of the opposite side.
      - family_split_rank1_vs_topk: per-family bars, dark rank-1 + light rest.

    role only affects labels ('source' = recurring damaging sources,
    'target' = recurring robust targets).
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    cfg = ensure_v2(config) if config is not None else AttacksV2Config()
    df = _select_summary_rows(summary, selected_entities=selected_entities,
                              sort_by=sort_by, max_rows=max_rows)
    if df.empty:
        raise ValueError("No rows to plot after selection/filtering.")
    df = df.iloc[::-1]  # horizontal bars: best at top
    n = len(df)
    opposite = "target" if role == "source" else "source"

    if variant == "family_split_rank1_vs_topk":
        if figsize is None:
            figsize = (8.5, max(3.2, 0.85 * n + 1.2))
        fig, ax = plt.subplots(figsize=figsize)
        bar_h = 0.26
        y = np.arange(n)
        for i, fam in enumerate(FAMILIES):
            offset = (i - 1) * bar_h
            total = df[f"topk_family_{fam.lower()}_count"].to_numpy(float)
            r1 = df[f"rank1_family_{fam.lower()}_count"].to_numpy(float)
            color = FAMILY_COLORS[fam]
            ax.barh(y + offset, total, height=bar_h, color=color, alpha=0.38)
            ax.barh(y + offset, r1, height=bar_h, color=color, alpha=0.95)
        ax.set_yticks(y)
        ax.set_yticklabels(df["entity"], fontsize=8.5)
        handles = [Patch(facecolor=FAMILY_COLORS[f], alpha=0.95, label=f"{f} {opposite}s (dark = rank 1)")
                   for f in FAMILIES]
        ax.legend(handles=handles, frameon=False, fontsize=8)
        ax.set_xlabel(f"Top-{cfg.top_k} appearances split by {opposite} family")
    else:
        if figsize is None:
            figsize = (8.0, max(2.8, 0.45 * n + 1.2))
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(n)
        if variant == "rank1_vs_topk":
            total = df["topk_count"].to_numpy(float)
            r1 = df["rank1_count"].to_numpy(float)
            colors = [FAMILY_COLORS.get(str(f), "#888888") for f in df["family"]]
            ax.barh(y, total, color=colors, alpha=0.38, label=f"top-{cfg.top_k} count")
            ax.barh(y, r1, color=colors, alpha=0.95, label="rank-1 count")
            handles = [Patch(facecolor="0.25", alpha=0.95, label="rank-1 count"),
                       Patch(facecolor="0.25", alpha=0.38, label=f"rest of top-{cfg.top_k}")]
            ax.legend(handles=handles, frameon=False, fontsize=8)
            ax.set_xlabel(f"Appearances in top-{cfg.top_k} (bar colored by own family)")
        elif variant == "rank_decomposition":
            left = np.zeros(n)
            shades = ["#1b3a5c", "#3f74a3", "#9cc1dd", "#d3e3f0", "#eef4f9"]
            for k in range(1, cfg.top_k + 1):
                col = f"rank{k}_count"
                if col not in df.columns:
                    continue
                vals = df[col].to_numpy(float)
                ax.barh(y, vals, left=left, color=shades[min(k - 1, len(shades) - 1)],
                        label=f"rank {k}")
                left += vals
            ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel(f"Appearances in top-{cfg.top_k} by rank")
        elif variant == "family_composition":
            left = np.zeros(n)
            for fam in FAMILIES:
                vals = df[f"topk_family_{fam.lower()}_count"].to_numpy(float)
                ax.barh(y, vals, left=left, color=FAMILY_COLORS[fam],
                        label=f"{fam} {opposite}s")
                left += vals
            ax.legend(frameon=False, fontsize=8)
            ax.set_xlabel(f"Top-{cfg.top_k} appearances by {opposite} family")
        else:
            raise ValueError(f"Unknown figure variant: {variant!r}")
        ax.set_yticks(y)
        ax.set_yticklabels(df["entity"], fontsize=8.5)

    role_label = "damaging sources" if role == "source" else "robust targets"
    ax.set_title(title or f"Recurring {role_label} ({variant})")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Directional asymmetry
# ---------------------------------------------------------------------------

def build_directional_pair_table(pair_df: pd.DataFrame, config: AttacksV2Config) -> pd.DataFrame:
    """A->B vs B->A metric for every unordered group pair with both directions.

    delta = a_to_b - b_to_a.  For damage-oriented metrics (e.g. Transfer ASR),
    positive delta means A attacks B more strongly than B attacks A.
    """
    cfg = ensure_v2(config)
    metric = resolve_metric(cfg.metric)
    grouped = aggregate_pairs_v2(pair_df, cfg)
    info = _group_info(pair_df, cfg).set_index("group")
    lut = {(str(r["source_group"]), str(r["target_group"])): float(r[metric])
           for _, r in grouped.iterrows()}
    groups = sorted({g for pair in lut for g in pair})

    rows: list[dict[str, object]] = []
    for a, b in itertools.combinations(groups, 2):
        ab, ba = lut.get((a, b)), lut.get((b, a))
        if ab is None or ba is None:
            continue
        rows.append({
            "group_a": a,
            "group_b": b,
            "family_a": info["family"].get(a, "Unknown"),
            "family_b": info["family"].get(b, "Unknown"),
            "architecture_a": info["architecture"].get(a, "Unknown"),
            "architecture_b": info["architecture"].get(b, "Unknown"),
            "a_to_b": ab,
            "b_to_a": ba,
            "delta": ab - ba,
            "abs_delta": abs(ab - ba),
            "metric": metric,
            "comparison_mode": cfg.comparison_mode,
            "source_grain": cfg.source_grain,
            "target_grain": cfg.target_grain,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if cfg.selected_rows:
        mask = _apply_selection_mask(table["group_a"], cfg.selected_rows)
        mask |= _apply_selection_mask(table["group_b"], cfg.selected_rows)
        table = table[mask]
    return table.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def plot_directional_asymmetry_bars(
    pair_table: pd.DataFrame,
    config: AttacksV2Config | None = None,
    *,
    top_n: int = 15,
    selected_pairs: Sequence[tuple[str, str]] | None = None,
    sort_bars: str = "delta",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Horizontal bars of delta = (A->B) - (B->A) for the most asymmetric pairs.

    The top_n pairs are picked by |delta|; bars are then ordered by ``sort_bars``:
    ``"delta"`` (signed, positives at the top) or ``"abs_delta"``.  Bars point
    right when A out-attacks B and left when B out-attacks A; each bar is
    labelled with its delta.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    df = pair_table.copy()
    if selected_pairs:
        wanted = {(str(a), str(b)) for a, b in selected_pairs}
        wanted |= {(b, a) for a, b in wanted}
        df = df[[(a, b) in wanted for a, b in zip(df["group_a"], df["group_b"])]]
    else:
        df = df.sort_values("abs_delta", ascending=False).head(top_n)
    if df.empty:
        raise ValueError("No pairs to plot.")
    if sort_bars not in {"delta", "abs_delta"}:
        raise ValueError(f"Unknown sort_bars: {sort_bars!r} (use 'delta' or 'abs_delta')")
    # Ascending sort puts the largest value at the top of the horizontal chart.
    df = df.sort_values(sort_bars, ascending=True)

    metric = str(df["metric"].iloc[0])
    n = len(df)
    if figsize is None:
        figsize = (8.8, max(3.0, 0.45 * n + 1.4))
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)
    deltas = df["delta"].to_numpy(float)
    colors = np.where(deltas >= 0, "#C44E52", "#4C72B0")
    ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0)

    labels = [f"{a}  →  {b}" for a, b in zip(df["group_a"], df["group_b"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    span = max(abs(deltas.min()), abs(deltas.max())) or 1.0
    for yi, d in zip(y, deltas):
        ax.annotate(f"{d:+.3f}", (d, yi), xytext=(4 if d >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if d >= 0 else "right", fontsize=7.5)
    ax.set_xlim(-1.25 * span, 1.25 * span)
    direction = "A attacks B harder" if metric_higher_is_damage(metric) else "B more robust to A"
    ax.set_xlabel(f"delta = (A→B) − (B→A) {metric}   (positive: {direction})")
    ax.set_title(title or f"Directional transfer asymmetry ({metric})")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Smoke test on synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_pair_metrics(seed: int = 0) -> pd.DataFrame:
    """Small synthetic pair-metrics table matching the real CSV schema."""
    rng = np.random.default_rng(seed)
    models = [
        ("Classical", "CNN", "LeNet"),
        ("Classical", "CNN", "Wide"),
        ("Classical", "MLP", "Base"),
        ("Classical", "ResNet18", "Base"),
        ("Hybrid", "HVQL", "ansatz=basic"),
        ("Hybrid", "HVQL", "ansatz=strong"),
        ("Hybrid", "HQuanv", "ansatz=basic"),
        ("Quantum", "VQM", "ansatz=basic"),
        ("Quantum", "VQM", "ansatz=random"),
        ("Quantum", "VQM", "ansatz=strong"),
    ]
    rows = []
    for sf, sa, sv in models:
        for tf, ta, tv in models:
            src = f"{sf}/{sa}/{sv}"
            tgt = f"{tf}/{ta}/{tv}"
            self_attack = src == tgt
            asr = float(rng.uniform(0.05, 0.7))
            rows.append({
                "Source Model": src, "Source Family": sf, "Source Architecture": sa,
                "Source Variant": sv,
                "Target Model": tgt, "Target Family": tf, "Target Architecture": ta,
                "Target Variant": tv,
                "WB/BB Flag": "WB" if self_attack else "BB",
                "Is Self Attack": self_attack,
                "Clean Accuracy": float(rng.uniform(0.85, 0.99)),
                "Accuracy Drop": asr * float(rng.uniform(0.5, 1.0)),
                "Transfer Robust Accuracy": np.nan if self_attack else 1.0 - asr,
                "Transfer ASR": np.nan if self_attack else asr,
                "Mean Confidence Drop": asr * float(rng.uniform(0.3, 0.9)),
                "Mean Margin Drop": asr * float(rng.uniform(1.0, 8.0)),
                "Flip Rate": asr * float(rng.uniform(0.8, 1.1)),
            })
    return pd.DataFrame(rows)


def run_smoke_test(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Verify scatter, matrix, top-k summaries and asymmetry build end-to-end."""
    pair_df = make_synthetic_pair_metrics()
    results: dict[str, pd.DataFrame] = {}

    cfg = get_v2_template("architecture_main", blacklist_contains=("HQuanv",))
    results["scatter"] = build_source_target_scatter(pair_df, cfg, display_mode="both")
    assert {"coarse", "fine"} <= set(results["scatter"]["level"]), "both-mode scatter missing a level"

    results["matrix"] = build_attack_matrix(pair_df, cfg)
    assert results["matrix"].shape[0] >= 2 and results["matrix"].shape[1] >= 2

    for mode in _RANKING_SPEC:
        results[f"topk_long_{mode}"] = build_topk_instances(pair_df, cfg, ranking_mode=mode)
        assert not results[f"topk_long_{mode}"].empty, f"empty top-k table for {mode}"
    summary, _ = build_recurring_damaging_sources(pair_df, cfg)
    results["recurring_damaging_sources"] = summary
    assert {"rank1_count", "topk_count", "topk_family_classical_count"} <= set(summary.columns)
    summary_t, _ = build_recurring_robust_targets(pair_df, cfg)
    results["recurring_robust_targets"] = summary_t

    results["asymmetry"] = build_directional_pair_table(pair_df, cfg)
    assert not results["asymmetry"].empty
    a, b = results["asymmetry"].iloc[0][["a_to_b", "b_to_a"]]
    d = results["asymmetry"].iloc[0]["delta"]
    assert abs((a - b) - d) < 1e-9

    # Ansatz split, family grain, and robust-accuracy direction also build.
    results["matrix_ansatz"] = build_attack_matrix(
        pair_df, get_v2_template("ansatz_split"))
    assert any("VQM-" in str(x) for x in results["matrix_ansatz"].index), "ansatz split not applied"
    results["scatter_family"] = build_source_target_scatter(
        pair_df, get_v2_template("family_synthesis"))
    robust_cfg = cfg.with_updates(metric="Transfer Robust Accuracy")
    long_robust = build_topk_instances(pair_df, robust_cfg, ranking_mode="robust_targets_for_sources")
    top1 = long_robust[long_robust["rank"] == 1]
    rest = long_robust[long_robust["rank"] > 1]
    if len(top1) and len(rest):
        assert top1["metric_value"].mean() >= rest["metric_value"].mean(), (
            "robust ranking direction wrong for Transfer Robust Accuracy")

    if verbose:
        for name, df in results.items():
            print(f"[smoke] {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        print("[smoke] all checks passed")
    return results


__all__ = [
    "AttacksV2Config",
    "V2_TEMPLATES",
    "ensure_v2",
    "get_v2_template",
    "list_v2_templates",
    "v2_output_dir",
    "metric_direction_notes",
    "aggregate_pairs_v2",
    "load_or_build_pair_metrics",
    "export_table",
    "export_figure",
    "save_figure",
    "build_source_target_scatter",
    "plot_source_target_scatter",
    "build_attack_matrix",
    "plot_attack_matrix",
    "build_topk_instances",
    "summarize_recurring",
    "build_recurring_damaging_sources",
    "build_recurring_robust_targets",
    "plot_topk_ranking",
    "build_directional_pair_table",
    "plot_directional_asymmetry_bars",
    "make_synthetic_pair_metrics",
    "run_smoke_test",
]
