"""Black-box transfer ASR scatter: outgoing vs incoming attack potential."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analytics.paths import find_project_root
from analytics.plots.qaml_scatter_style import (
    draw_quadrant_region_annotations,
    enrich_plot_schema,
    render_scatter_chart,
)
from analytics.attacks.attack_matrix import ATTACK_TABLE_METRICS  # noqa: F401 — re-exported
from analytics.tables.extract_attack import build_pair_metrics
from analytics.tables.model_blacklist import MODEL_BLACKLIST

log = logging.getLogger(__name__)

TransferFilterMode = Literal[
    "inter_model",
    "inter_architecture",
    "inter_family",
    "attack_lens_classical",
    "attack_lens_hybrid",
    "attack_lens_quantum",
]

ATTACK_LENS_MODES: dict[str, str] = {
    "attack_lens_classical": "Classical",
    "attack_lens_hybrid": "Hybrid",
    "attack_lens_quantum": "Quantum",
}

THESIS_ARCHITECTURES: tuple[str, ...] = (
    "CNN",
    "ResNet18",
    "MLPMixer",
    "MLP",
    "VQM",
    "HVQM",
    "HVQL",
    "HPVQL",
    "HQuanv",
    "HQE",
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
    "hpvql": "HPVQL",
    "hpvqm": "HPVQL",
    "hquanv": "HQuanv",
    "hqe": "HQE",
    "drophvql": "HVQL",
    "reshvql": "HVQL",
}

# Profile columns used for scatter x/y (outgoing = source, incoming = target).
OUTGOING_COL = "outgoing_score"
INCOMING_COL = "incoming_score"
WHITEBOX_COL = "whitebox_score"

# attack_table metric → (black-box pair_df column, white-box column, plot title fragment)
PAIR_METRIC_MAP: dict[str, tuple[str, str, str]] = {
    "ASR": ("Transfer ASR", "ASR", "ASR"),
    "robust_accuracy": ("Transfer Robust Accuracy", "Robust Accuracy", "Robust accuracy"),
    "flip_rate": ("Flip Rate", "Flip Rate", "Flip rate"),
    "mean_confidence_drop_all": ("Mean Confidence Drop", "Mean Confidence Drop", "Confidence drop"),
    "mean_margin_drop_all": ("Mean Margin Drop", "Mean Margin Drop", "Margin drop"),
    "mean_confidence_drop_cc": ("Mean Confidence Drop", "Mean Confidence Drop", "Confidence drop (CC)"),
    "mean_margin_drop_cc": ("Mean Margin Drop", "Mean Margin Drop", "Margin drop (CC)"),
    "transfer_gap": ("Transfer Gap", "Transfer Gap", "Transfer gap"),
}

# Back-compat aliases
PAIR_METRIC_COL = PAIR_METRIC_MAP["ASR"][0]
WB_METRIC_COL = PAIR_METRIC_MAP["ASR"][1]

DEFAULT_AXIS_CLIP = (0.4, 1.0)

FAMILY_FACE_COLORS: dict[str, str] = {
    "Classical": "#4C72B0",
    "Hybrid": "#C44E52",
    "Quantum": "#55A868",
}

ARCH_MARKERS: dict[str, str] = {
    "CNN": "o",
    "ResNet18": "s",
    "MLPMixer": "D",
    "MLP": "^",
    "VQM": "o",
    "HVQM": "s",
    "HVQL": "D",
    "HPVQL": "v",
    "HQuanv": "P",
    "HQE": "*",
}

ARCH_FIGURE_DPI = 220

METRIC_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "ASR": (
        "Attack potential (mean ASR as source)",
        "Attack vulnerability (mean ASR as target)",
    ),
    "robust_accuracy": (
        "Robust accuracy of attacked targets (as source)",
        "Robust accuracy under transfer attack (as target)",
    ),
}

ATTACK_LENS_AXIS_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "ASR": {
        "Classical": (
            "Attack potential on Classical targets (mean ASR as source)",
            "Attack vulnerability from Classical sources (mean ASR as target)",
        ),
        "Hybrid": (
            "Attack potential on Hybrid targets (mean ASR as source)",
            "Attack vulnerability from Hybrid sources (mean ASR as target)",
        ),
        "Quantum": (
            "Attack potential on Quantum targets (mean ASR as source)",
            "Attack vulnerability from Quantum sources (mean ASR as target)",
        ),
    },
    "robust_accuracy": {
        "Classical": (
            "Robust accuracy of Classical targets when attacked (as source)",
            "Robust accuracy under Classical transfer attacks (as target)",
        ),
        "Hybrid": (
            "Robust accuracy of Hybrid targets when attacked (as source)",
            "Robust accuracy under Hybrid transfer attacks (as target)",
        ),
        "Quantum": (
            "Robust accuracy of Quantum targets when attacked (as source)",
            "Robust accuracy under Quantum transfer attacks (as target)",
        ),
    },
}

METRIC_QUADRANT_REGION_ANNOTATIONS: dict[str, dict[str, str]] = {
    "ASR": {
        "TR": (
            "Strong transferable source\nTransfer-vulnerable target"
        ),
        "BR": (
            "Strong transferable source\nTransfer-resistant target"
        ),
        "TL": (
            "Weak transferable source\nTransfer-vulnerable target"
        ),
        "BL": (
            "Weak transferable source\nTransfer-resistant target"
        ),
    },
    "robust_accuracy": {
        "TR": (
            "Mild transferable harm\nRobust under transfer"
        ),
        "BR": (
            "Mild transferable harm\nVulnerable under transfer"
        ),
        "TL": (
            "Severe transferable harm\nRobust under transfer"
        ),
        "BL": (
            "Severe transferable harm\nVulnerable under transfer"
        ),
    },
}

METRIC_QUADRANT_LABELS: dict[str, dict[str, str]] = {
    "ASR": {
        "TR": "Strong source, transfer-vulnerable target",
        "BR": "Strong source, transfer-resistant target",
        "TL": "Weak source, transfer-vulnerable target",
        "BL": "Weak source, transfer-resistant target",
    },
    "robust_accuracy": {
        "TR": "Mild transferable source, robust transfer target",
        "BR": "Mild transferable source, transfer-vulnerable target",
        "TL": "Severe transferable source, robust transfer target",
        "BL": "Severe transferable source, transfer-vulnerable target",
    },
}

# Back-compat aliases (ASR defaults).
X_LABEL = METRIC_AXIS_LABELS["ASR"][0]
Y_LABEL = METRIC_AXIS_LABELS["ASR"][1]
QUADRANT_REGION_ANNOTATIONS = METRIC_QUADRANT_REGION_ANNOTATIONS["ASR"]


@dataclass(frozen=True)
class TransferMetricSpec:
    """Resolved metric columns in ``pair_df`` from an attack-table metric name."""

    attack_metric: str
    bb_column: str
    wb_column: str
    display_name: str


@dataclass(frozen=True)
class TransferScatterPlotOptions:
    """Shared styling controls for variant and architecture transfer scatters."""

    metric: str = "ASR"
    xlim: tuple[float, float] = DEFAULT_AXIS_CLIP
    ylim: tuple[float, float] = DEFAULT_AXIS_CLIP
    clip_axes: bool = True
    show_quadrant_annotations: bool = True
    show_point_labels: bool = True


def available_transfer_metrics(pair_df: pd.DataFrame) -> list[str]:
    """Attack-table metric names usable for transfer scatter given ``pair_df`` columns."""
    available: list[str] = []
    for attack_metric, (bb_col, wb_col, _) in PAIR_METRIC_MAP.items():
        if bb_col in pair_df.columns or wb_col in pair_df.columns:
            available.append(attack_metric)
    return available


METRIC_ALIASES: dict[str, str] = {
    "robust accuracy": "robust_accuracy",
    "robust_acc": "robust_accuracy",
}


def resolve_transfer_metric(pair_df: pd.DataFrame, metric: str) -> TransferMetricSpec:
    """Map attack-matrix metric name to pair-metrics columns."""
    metric = METRIC_ALIASES.get(metric.strip().lower(), metric.strip())
    if metric not in ATTACK_TABLE_METRICS or metric not in PAIR_METRIC_MAP:
        allowed = ", ".join(PAIR_METRIC_MAP)
        raise ValueError(f"Unknown transfer metric {metric!r}. Choose one of: {allowed}")

    bb_pref, wb_pref, display = PAIR_METRIC_MAP[metric]
    bb_candidates = [bb_pref, wb_pref] if bb_pref != wb_pref else [bb_pref]
    bb_col = next((c for c in bb_candidates if c in pair_df.columns), None)
    if bb_col is None:
        raise KeyError(
            f"Metric {metric!r} needs one of {bb_candidates} in pair_df; "
            f"have: {list(pair_df.columns)}"
        )
    # Prefer column with most non-NaN black-box values
    bb_only = pair_df[~pair_df["Is Self Attack"].astype(bool)] if "Is Self Attack" in pair_df.columns else pair_df
    if len(bb_candidates) > 1:
        best = bb_col
        best_count = int(bb_only[bb_col].notna().sum())
        for cand in bb_candidates:
            n = int(bb_only[cand].notna().sum())
            if n > best_count:
                best, best_count = cand, n
        bb_col = best
    wb_col = wb_pref if wb_pref in pair_df.columns else bb_col
    return TransferMetricSpec(
        attack_metric=metric,
        bb_column=bb_col,
        wb_column=wb_col,
        display_name=display,
    )


def metric_file_slug(metric: str) -> str:
    return metric.replace(" ", "_").replace("/", "_")


def _resolve_plot_metric(
    metric: str | None = None,
    df: pd.DataFrame | None = None,
) -> str:
    if metric is not None:
        return METRIC_ALIASES.get(metric.strip().lower(), metric.strip())
    if df is not None and "metric" in df.columns and len(df):
        return str(df["metric"].iloc[0])
    return "ASR"


def attack_lens_family(mode: str | None) -> str | None:
    if mode is None:
        return None
    return ATTACK_LENS_MODES.get(mode)


def axis_labels_for_metric(
    metric: str,
    *,
    filter_mode: str | None = None,
) -> tuple[str, str]:
    key = _resolve_plot_metric(metric)
    lens = attack_lens_family(filter_mode)
    if lens is not None:
        by_metric = ATTACK_LENS_AXIS_LABELS.get(key, ATTACK_LENS_AXIS_LABELS["ASR"])
        return by_metric[lens]
    return METRIC_AXIS_LABELS.get(key, METRIC_AXIS_LABELS["ASR"])


def quadrant_region_annotations_for_metric(metric: str) -> dict[str, str]:
    key = _resolve_plot_metric(metric)
    return METRIC_QUADRANT_REGION_ANNOTATIONS.get(
        key, METRIC_QUADRANT_REGION_ANNOTATIONS["ASR"]
    )


def quadrant_labels_for_metric(metric: str) -> dict[str, str]:
    key = _resolve_plot_metric(metric)
    return METRIC_QUADRANT_LABELS.get(key, METRIC_QUADRANT_LABELS["ASR"])


def parse_canonical_model_name(model_name: str) -> tuple[str, str, str]:
    """Mirror ``attack_matrix.parse_canonical_model_name`` (avoids heavy package init)."""
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


def load_pair_metrics(
    attack_dir: Path,
    *,
    output_path: Path | None = None,
    model_blacklist: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build seed-aggregated pair metrics from attack experiment tables."""
    return build_pair_metrics(
        attack_dir=attack_dir,
        output_path=output_path,
        model_blacklist=model_blacklist or MODEL_BLACKLIST,
    )


def normalize_architecture(arch: str | None) -> str | None:
    if arch is None or not str(arch).strip():
        return None
    raw = str(arch).strip()
    key = raw.lower()
    return ARCH_NORMALIZE.get(key, raw)


def parse_ansatz(variant: str) -> str:
    if not isinstance(variant, str) or not variant.strip():
        return "none"
    m = re.search(r"ansatz=([^,\s]+)", variant, re.IGNORECASE)
    if not m:
        return "none"
    return m.group(1).lower().strip()


def parse_qdepth(variant: str) -> float:
    if not isinstance(variant, str) or not variant.strip():
        return float("nan")
    m = re.search(r"qdepth=(\d+)", variant, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"n_qdepth=(\d+)", variant, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return float("nan")


def _metric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _black_box_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Black-box transfer pairs excluding self-attacks."""
    work = pair_df.copy()
    if "Is Self Attack" in work.columns:
        work = work[~work["Is Self Attack"].astype(bool)]
    elif "WB/BB Flag" in work.columns:
        work = work[work["WB/BB Flag"] == "BB"]
    return work[work["Source Model"] != work["Target Model"]]


def _attach_metric_values(
    work: pd.DataFrame,
    metric_spec: TransferMetricSpec,
) -> pd.DataFrame:
    work = work.copy()
    work["_metric_value"] = _metric_series(work, metric_spec.bb_column)
    if work["_metric_value"].isna().all() and metric_spec.bb_column != metric_spec.wb_column:
        work["_metric_value"] = _metric_series(work, metric_spec.wb_column)
    return work.dropna(subset=["_metric_value"])


def filter_transfer_pairs(
    pair_df: pd.DataFrame,
    mode: TransferFilterMode,
    metric_spec: TransferMetricSpec,
) -> pd.DataFrame:
    """Keep black-box pairs for the requested transfer scope."""
    if mode in ATTACK_LENS_MODES:
        raise ValueError(
            f"Mode {mode!r} uses asymmetric pair filters; use compute_model_transfer_profile instead."
        )

    work = _black_box_pairs(pair_df)

    if mode == "inter_model":
        pass
    elif mode == "inter_architecture":
        src_arch = work["Source Architecture"].map(
            lambda a: normalize_architecture(str(a)) or str(a)
        )
        tgt_arch = work["Target Architecture"].map(
            lambda a: normalize_architecture(str(a)) or str(a)
        )
        work = work[src_arch != tgt_arch]
    elif mode == "inter_family":
        work = work[
            work["Source Family"].astype(str) != work["Target Family"].astype(str)
        ]
    else:
        raise ValueError(f"Unknown transfer filter mode: {mode!r}")

    return _attach_metric_values(work, metric_spec)


def _attack_lens_pair_slices(
    pair_df: pd.DataFrame,
    lens_family: str,
    metric_spec: TransferMetricSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outgoing slice: any source attacking targets in *lens_family*.
    Incoming slice: sources in *lens_family* attacking any target.
    """
    bb = _black_box_pairs(pair_df)
    family = str(lens_family)
    outgoing_bb = _attach_metric_values(
        bb[bb["Target Family"].astype(str) == family],
        metric_spec,
    )
    incoming_bb = _attach_metric_values(
        bb[bb["Source Family"].astype(str) == family],
        metric_spec,
    )
    return outgoing_bb, incoming_bb


def _model_metadata(model_name: str) -> dict[str, object]:
    family, arch, variant = parse_canonical_model_name(model_name)
    arch_norm = normalize_architecture(arch)
    return {
        "model": model_name,
        "family": family,
        "architecture": arch_norm or arch,
        "variant": variant,
        "ansatz": parse_ansatz(variant),
        "qdepth": parse_qdepth(variant),
    }


def _secondary_means(group: pd.DataFrame) -> dict[str, float]:
    cols = {
        "transfer_robust_accuracy": "Transfer Robust Accuracy",
        "accuracy_drop": "Accuracy Drop",
        "mean_confidence_drop": "Mean Confidence Drop",
        "mean_margin_drop": "Mean Margin Drop",
    }
    out: dict[str, float] = {}
    for key, col in cols.items():
        if col in group.columns:
            out[key] = float(pd.to_numeric(group[col], errors="coerce").mean())
        else:
            out[key] = float("nan")
    return out


def compute_model_transfer_profile(
    pair_df: pd.DataFrame,
    mode: TransferFilterMode,
    metric: str = "ASR",
) -> pd.DataFrame:
    """Per-configuration outgoing/incoming transfer metric (AUC) and secondary fields."""
    spec = resolve_transfer_metric(pair_df, metric)
    lens_family = attack_lens_family(mode)
    if lens_family is not None:
        outgoing_bb, incoming_bb = _attack_lens_pair_slices(pair_df, lens_family, spec)
    else:
        bb = filter_transfer_pairs(pair_df, mode, spec)
        outgoing_bb = incoming_bb = bb

    outgoing_rows: list[dict[str, object]] = []
    for source, group in outgoing_bb.groupby("Source Model", sort=False):
        meta = _model_metadata(str(source))
        sec = _secondary_means(group)
        outgoing_rows.append(
            {
                **meta,
                OUTGOING_COL: float(group["_metric_value"].mean()),
                "n_source_pairs": int(len(group)),
                **sec,
            }
        )
    outgoing_df = pd.DataFrame(outgoing_rows)

    incoming_rows: list[dict[str, object]] = []
    for target, group in incoming_bb.groupby("Target Model", sort=False):
        meta = _model_metadata(str(target))
        incoming_rows.append(
            {
                "model": meta["model"],
                INCOMING_COL: float(group["_metric_value"].mean()),
                "n_target_pairs": int(len(group)),
            }
        )
    incoming_df = pd.DataFrame(incoming_rows)

    wb = pair_df[pair_df["Is Self Attack"].astype(bool)].copy()
    if wb.empty and "WB/BB Flag" in pair_df.columns:
        wb = pair_df[pair_df["WB/BB Flag"] == "WB"]
    wb_vals = _metric_series(wb, spec.wb_column)
    wb_score = (
        wb.assign(**{spec.wb_column: wb_vals})
        .groupby("Source Model", sort=False)[spec.wb_column]
        .mean()
        .reset_index()
        .rename(columns={"Source Model": "model", spec.wb_column: WHITEBOX_COL})
    )

    if outgoing_df.empty:
        return pd.DataFrame()

    profile = outgoing_df.merge(incoming_df, on="model", how="outer")
    profile = profile.merge(wb_score, on="model", how="left")
    profile["transfer_gap"] = profile[WHITEBOX_COL] - profile[OUTGOING_COL]
    profile["metric"] = spec.attack_metric
    profile["metric_display"] = spec.display_name
    profile["filter_mode"] = mode
    if lens_family is not None:
        profile["attack_lens_family"] = lens_family
    return profile.sort_values(["family", "architecture", "model"]).reset_index(drop=True)


def aggregate_architecture_profile(model_df: pd.DataFrame) -> pd.DataFrame:
    """Mean across configurations per thesis architecture."""
    if model_df.empty:
        return pd.DataFrame()

    work = model_df.copy()
    work["architecture"] = work["architecture"].map(
        lambda a: normalize_architecture(str(a)) if pd.notna(a) else a
    )
    allowed = set(THESIS_ARCHITECTURES)
    skipped = sorted(
        {a for a in work["architecture"].dropna().unique() if a not in allowed}
    )
    if skipped:
        log.warning("Skipping architectures outside thesis set: %s", skipped)
    work = work[work["architecture"].isin(allowed)]

    numeric_cols = [
        OUTGOING_COL,
        INCOMING_COL,
        WHITEBOX_COL,
        "transfer_gap",
        "transfer_robust_accuracy",
        "accuracy_drop",
        "mean_confidence_drop",
        "mean_margin_drop",
    ]
    count_cols = ["n_source_pairs", "n_target_pairs"]

    agg: dict[str, str] = {c: "mean" for c in numeric_cols if c in work.columns}
    for c in count_cols:
        if c in work.columns:
            agg[c] = "sum"

    arch_df = (
        work.groupby(["family", "architecture"], dropna=False)
        .agg({**agg, "model": "count"})
        .rename(columns={"model": "n_configs"})
        .reset_index()
    )
    arch_df = arch_df.rename(columns={"family": "Family", "architecture": "Architecture"})
    if "metric" in work.columns:
        arch_df["metric"] = work["metric"].iloc[0]
    if "metric_display" in work.columns:
        arch_df["metric_display"] = work["metric_display"].iloc[0]
    order = {a: i for i, a in enumerate(THESIS_ARCHITECTURES)}
    arch_df["_order"] = arch_df["Architecture"].map(order)
    return arch_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def assign_quadrant(
    outgoing: float,
    incoming: float,
    x_mean: float,
    y_mean: float,
) -> str:
    high_x = outgoing >= x_mean
    high_y = incoming >= y_mean
    if high_x and high_y:
        return "TR"
    if high_x and not high_y:
        return "BR"
    if not high_x and high_y:
        return "TL"
    return "BL"


QUADRANT_LABELS = METRIC_QUADRANT_LABELS["ASR"]


def add_quadrants(df: pd.DataFrame, *, metric: str | None = None) -> pd.DataFrame:
    """Add quadrant column using global means of outgoing/incoming."""
    plot_metric = _resolve_plot_metric(metric, df)
    label_map = quadrant_labels_for_metric(plot_metric)
    out = df.copy()
    x_mean = float(out[OUTGOING_COL].mean())
    y_mean = float(out[INCOMING_COL].mean())
    out["x_mean"] = x_mean
    out["y_mean"] = y_mean
    out["quadrant"] = [
        assign_quadrant(x, y, x_mean, y_mean)
        for x, y in zip(out[OUTGOING_COL], out[INCOMING_COL])
    ]
    out["quadrant_label"] = out["quadrant"].map(label_map)
    return out


def detect_outliers(
    df: pd.DataFrame,
    *,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Flag points far from the global mean in (outgoing, incoming) space."""
    if df.empty:
        return df.copy()

    x = df["outgoing_score"].to_numpy(dtype=float)
    y = df["incoming_score"].to_numpy(dtype=float)
    x_mean, y_mean = float(np.nanmean(x)), float(np.nanmean(y))
    dist = np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2)
    q1, q3 = np.nanpercentile(dist, [25, 75])
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr if iqr > 0 else q3

    out = df.copy()
    out["distance_from_mean"] = dist
    out["is_outlier"] = dist > threshold
    return out


def _resolve_project_root(out_path: Path, project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root
    try:
        return find_project_root(out_path)
    except FileNotFoundError:
        return out_path.resolve().parents[2]


def _filter_mode_label(mode: TransferFilterMode) -> str:
    if mode in ATTACK_LENS_MODES:
        return f"{ATTACK_LENS_MODES[mode]} attack view"
    return {
        "inter_model": "inter-model",
        "inter_architecture": "inter-architecture",
        "inter_family": "inter-family",
    }[mode]


def _mode_title(
    filter_mode: TransferFilterMode,
    level: str,
    metric_display: str,
    suffix: str = "",
) -> str:
    title = f"Transfer {metric_display} by {level} ({_filter_mode_label(filter_mode)})"
    return f"{title} — {suffix}" if suffix else title


def _effective_axis_limits(
    x: pd.Series,
    y: pd.Series,
    opts: TransferScatterPlotOptions,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Apply clip limits; if the configured window hides all data (e.g. robust_accuracy
    with default 40–100% clip), fall back to a padded data range.
    """
    if not opts.clip_axes:
        pad = 0.04
        return (
            (max(0.0, float(x.min()) - pad), min(1.0, float(x.max()) + pad)),
            (max(0.0, float(y.min()) - pad), min(1.0, float(y.max()) + pad)),
        )

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    xlo, xhi = opts.xlim
    ylo, yhi = opts.ylim
    x_visible = x_max >= xlo and x_min <= xhi
    y_visible = y_max >= ylo and y_min <= yhi
    if x_visible and y_visible:
        return opts.xlim, opts.ylim

    pad = 0.04
    log.warning(
        "Axis clip x=%s y=%s hides %s data (x=[%.3f, %.3f], y=[%.3f, %.3f]); using data range.",
        opts.xlim,
        opts.ylim,
        opts.metric,
        x_min,
        x_max,
        y_min,
        y_max,
    )
    return (
        (max(0.0, x_min - pad), min(1.0, x_max + pad)),
        (max(0.0, y_min - pad), min(1.0, y_max + pad)),
    )


def _prepare_variant_plot_df(
    model_df: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    work = add_quadrants(model_df)
    return enrich_plot_schema(
        work,
        project_root=project_root,
        x_col=OUTGOING_COL,
        y_col=INCOMING_COL,
    )


def _apply_axis_limits(
    ax: plt.Axes,
    *,
    clip_axes: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_data: pd.Series,
    y_data: pd.Series,
) -> None:
    if clip_axes:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
    else:
        x_max = float(x_data.max())
        y_max = float(y_data.max())
        ax.set_xlim(0, min(1.0, max(0.15, x_max + 0.04)))
        ax.set_ylim(0, min(1.0, max(0.15, y_max + 0.04)))


def plot_architecture_transfer_scatter(
    arch_df: pd.DataFrame,
    *,
    filter_mode: TransferFilterMode,
    out_path: Path,
    title_suffix: str = "",
    plot_options: TransferScatterPlotOptions | None = None,
):
    """Architecture-level scatter: family fill colour, architecture marker shape."""
    opts = plot_options or TransferScatterPlotOptions()
    if "metric_display" in arch_df.columns and len(arch_df):
        metric_display = str(arch_df["metric_display"].iloc[0])
    else:
        metric_display = PAIR_METRIC_MAP.get(opts.metric, ("", "", opts.metric))[2]
    data = add_quadrants(arch_df)
    if data.empty:
        raise ValueError("No architecture points to plot.")

    x_mean = float(data["x_mean"].iloc[0])
    y_mean = float(data["y_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in data.iterrows():
        family = str(row.get("Family", row.get("family", "Unknown")))
        arch = str(row["Architecture"])
        color = FAMILY_FACE_COLORS.get(family, "#888888")
        marker = ARCH_MARKERS.get(arch, "o")
        ax.scatter(
            row["outgoing_score"],
            row["incoming_score"],
            s=120,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.85,
            zorder=3,
        )
        if opts.show_point_labels:
            ax.annotate(
                arch,
                (row["outgoing_score"], row["incoming_score"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color="0.35",
            )

    xlim, ylim = _effective_axis_limits(
        data["outgoing_score"],
        data["incoming_score"],
        opts,
    )
    _apply_axis_limits(
        ax,
        clip_axes=True,
        xlim=xlim,
        ylim=ylim,
        x_data=data["outgoing_score"],
        y_data=data["incoming_score"],
    )
    x_label, y_label = axis_labels_for_metric(opts.metric, filter_mode=filter_mode)
    region_labels = quadrant_region_annotations_for_metric(opts.metric)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(_mode_title(filter_mode, "architecture", metric_display, title_suffix))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.axvline(x_mean, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(y_mean, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    if opts.show_quadrant_annotations:
        draw_quadrant_region_annotations(
            ax,
            x_mean,
            y_mean,
            labels=region_labels,
        )

    legend_handles = [
        Patch(facecolor=FAMILY_FACE_COLORS[f], edgecolor="white", label=f)
        for f in ("Classical", "Hybrid", "Quantum")
        if f in FAMILY_FACE_COLORS
    ]
    ax.legend(handles=legend_handles, title="Family", loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=ARCH_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_variant_transfer_scatter(
    model_df: pd.DataFrame,
    *,
    filter_mode: TransferFilterMode,
    out_path: Path,
    title_suffix: str = "",
    project_root: Path | None = None,
    plot_options: TransferScatterPlotOptions | None = None,
):
    """Configuration-level scatter using QAML white-box figure styling."""
    opts = plot_options or TransferScatterPlotOptions()
    root = _resolve_project_root(out_path, project_root)
    if "metric_display" in model_df.columns and len(model_df):
        metric_display = str(model_df["metric_display"].iloc[0])
    else:
        metric_display = PAIR_METRIC_MAP.get(opts.metric, ("", "", opts.metric))[2]
    plot_df = _prepare_variant_plot_df(model_df, root)
    if plot_df.empty:
        raise ValueError("No configuration points to plot after enrichment.")

    x_mean = float(plot_df["x_mean"].iloc[0])
    y_mean = float(plot_df["y_mean"].iloc[0])

    xlim, ylim = _effective_axis_limits(
        plot_df["_plot_x"],
        plot_df["_plot_y"],
        opts,
    )
    x_label, y_label = axis_labels_for_metric(opts.metric, filter_mode=filter_mode)
    region_labels = quadrant_region_annotations_for_metric(opts.metric)
    return render_scatter_chart(
        plot_df,
        out_path=out_path,
        xlabel=x_label,
        ylabel=y_label,
        title=_mode_title(filter_mode, "configuration", metric_display, title_suffix),
        show_labels=opts.show_point_labels,
        quadrant_means=(x_mean, y_mean),
        show_quadrant_annotations=opts.show_quadrant_annotations,
        quadrant_annotation_labels=region_labels,
        clip_axes=True,
        xlim=xlim,
        ylim=ylim,
    )


def build_transfer_summary_table(arch_df: pd.DataFrame) -> pd.DataFrame:
    """Summary table for architecture-level transfer metrics."""
    plot_metric = _resolve_plot_metric(df=arch_df)
    label_map = quadrant_labels_for_metric(plot_metric)
    data = add_quadrants(arch_df, metric=plot_metric)
    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        arch = str(row["Architecture"])
        note_parts: list[str] = []
        q = str(row["quadrant"])
        note_parts.append(label_map.get(q, q))
        if arch == "HQE":
            out_v = row["outgoing_score"]
            in_v = row["incoming_score"]
            if plot_metric == "robust_accuracy":
                if out_v < row["x_mean"] and in_v > row["y_mean"]:
                    note_parts.append(
                        "HQE: severe transferable harm as source, robust under transfer as target"
                    )
                elif in_v < row["y_mean"] and out_v > row["x_mean"]:
                    note_parts.append(
                        "HQE: vulnerable under transfer despite mild outgoing harm"
                    )
            else:
                if out_v > row["x_mean"] and in_v < row["y_mean"]:
                    note_parts.append("HQE: strong as source, transfer-resistant as target")
                elif in_v > row["y_mean"] and out_v < row["x_mean"]:
                    note_parts.append("HQE: vulnerable as target despite weak outgoing")
        gap = row.get("transfer_gap", float("nan"))
        if pd.notna(gap) and gap > 0.15:
            note_parts.append("large white-box vs transfer gap")

        rows.append(
            {
                "Architecture": arch,
                "Family": row.get("Family", row.get("family")),
                "n_configs": int(row.get("n_configs", 0)),
                "outgoing_score": row["outgoing_score"],
                "incoming_score": row["incoming_score"],
                "whitebox_score": row.get("whitebox_score"),
                "transfer_gap": gap,
                "n_source_pairs": row.get("n_source_pairs"),
                "n_target_pairs": row.get("n_target_pairs"),
                "quadrant": q,
                "note": "; ".join(note_parts),
            }
        )
    return pd.DataFrame(rows)


def generate_interpretation_notes(
    arch_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    metric: str | None = None,
) -> str:
    """Markdown interpretation for thesis-style bullet questions."""
    plot_metric = _resolve_plot_metric(metric, arch_df)
    arch = add_quadrants(arch_df, metric=plot_metric)
    x_mean = float(arch["outgoing_score"].mean())
    y_mean = float(arch["incoming_score"].mean())
    x_std = float(arch["outgoing_score"].std(ddof=0))
    y_std = float(arch["incoming_score"].std(ddof=0))
    wb_median = float(arch["whitebox_score"].median()) if "whitebox_score" in arch.columns else float("nan")

    if plot_metric == "robust_accuracy":
        strong_source = arch[arch["outgoing_score"] < x_mean - 0.5 * x_std]
        resistant_target = arch[arch["incoming_score"] > y_mean]
        weak_source_high_wb = arch[
            (arch["whitebox_score"] >= wb_median)
            & (arch["outgoing_score"] > x_mean)
        ]
        vuln_despite_weak_out = arch[
            (arch["outgoing_score"] > x_mean)
            & (arch["incoming_score"] < y_mean)
        ]
        title = "# Black-box transfer robust accuracy interpretation"
        terminology = [
            "Terminology: high incoming transfer robust accuracy indicates **target-side",
            "transfer resistance** (accuracy retained under cross-model attacks). Low outgoing",
            "transfer robust accuracy indicates **severe transferable harm** (targets lose more",
            "accuracy when this model is the attack source). White-box robust accuracy is reported",
            "separately and must not be merged with transfer metrics.",
        ]
        strong_source_heading = "## Severe transferable sources (low outgoing robust accuracy)"
        resistant_heading = "## Transfer-resistant targets (high incoming robust accuracy)"
        vuln_heading = "## Vulnerable targets despite mild outgoing transferable harm"
        weak_wb_heading = (
            "## Mild transferable harm despite high white-box robust accuracy "
            "(high WB, high outgoing robust accuracy)"
        )
    else:
        strong_source = arch[arch["outgoing_score"] > x_mean + 0.5 * x_std]
        resistant_target = arch[arch["incoming_score"] < y_mean]
        weak_source_high_wb = arch[
            (arch["whitebox_score"] >= wb_median)
            & (arch["outgoing_score"] < x_mean)
        ]
        vuln_despite_weak_out = arch[
            (arch["outgoing_score"] < x_mean)
            & (arch["incoming_score"] > y_mean)
        ]
        title = "# Black-box transfer ASR interpretation"
        terminology = [
            "Terminology: low incoming ASR indicates **target-side transfer resistance**, not",
            "general robustness. Low outgoing ASR indicates **weak transferable source potential**,",
            "not model robustness. White-box ASR is reported separately and must not be merged",
            "with transfer metrics.",
        ]
        strong_source_heading = "## Strong transferable sources (high outgoing ASR)"
        resistant_heading = "## Transfer-resistant targets (low incoming ASR)"
        vuln_heading = "## Vulnerable targets despite weak outgoing attacks"
        weak_wb_heading = "## Weak sources despite white-box stability (high WB, low outgoing)"

    lines = [
        title,
        "",
        *terminology,
        "",
        f"Global means: outgoing = {x_mean:.4f}, incoming = {y_mean:.4f}.",
        "",
        strong_source_heading,
        "",
    ]
    if strong_source.empty:
        if plot_metric == "robust_accuracy":
            lines.append("- None below mean − 0.5σ in this slice.")
        else:
            lines.append("- None above mean + 0.5σ in this slice.")
    else:
        for _, r in strong_source.iterrows():
            lines.append(
                f"- **{r['Architecture']}** ({r['Family']}): "
                f"outgoing={r['outgoing_score']:.4f}, "
                f"incoming={r['incoming_score']:.4f}, "
                f"quadrant={r['quadrant']}"
            )

    lines.extend(["", resistant_heading, ""])
    if resistant_target.empty:
        if plot_metric == "robust_accuracy":
            lines.append("- None above global incoming mean.")
        else:
            lines.append("- None below global incoming mean.")
    else:
        sort_asc = plot_metric != "robust_accuracy"
        sorted_rows = resistant_target.sort_values(
            "incoming_score", ascending=sort_asc
        )
        for _, r in sorted_rows.iterrows():
            lines.append(
                f"- **{r['Architecture']}**: incoming={r['incoming_score']:.4f} "
                f"(outgoing={r['outgoing_score']:.4f})"
            )

    lines.extend(["", weak_wb_heading, ""])
    if weak_source_high_wb.empty:
        lines.append("- No architecture in this category.")
    else:
        for _, r in weak_source_high_wb.iterrows():
            lines.append(
                f"- **{r['Architecture']}**: white-box={r['whitebox_score']:.4f}, "
                f"outgoing={r['outgoing_score']:.4f}, "
                f"transfer_gap={r['transfer_gap']:.4f}"
            )

    lines.extend(["", vuln_heading, ""])
    if vuln_despite_weak_out.empty:
        lines.append("- No architecture in this category.")
    else:
        for _, r in vuln_despite_weak_out.iterrows():
            lines.append(
                f"- **{r['Architecture']}**: outgoing={r['outgoing_score']:.4f}, "
                f"incoming={r['incoming_score']:.4f}"
            )

    lines.extend(["", "## HQE (source vs target roles)", ""])
    hqe_arch = arch[arch["Architecture"] == "HQE"]
    if hqe_arch.empty:
        lines.append("- HQE not present in architecture aggregate.")
    else:
        r = hqe_arch.iloc[0]
        lines.append(
            f"- HQE aggregate: outgoing={r['outgoing_score']:.4f} "
            f"(source potential), incoming={r['incoming_score']:.4f} "
            f"(target vulnerability), white-box={r.get('whitebox_score', float('nan')):.4f}."
        )
    hqe_models = model_df[model_df["architecture"] == "HQE"] if not model_df.empty else pd.DataFrame()
    if not hqe_models.empty:
        for _, r in hqe_models.iterrows():
            lines.append(
                f"- Config `{r['model']}`: outgoing={r['outgoing_score']:.4f}, "
                f"incoming={r['incoming_score']:.4f}."
            )

    lines.extend(["", "## Family / mechanism patterns", ""])
    for family in ("Classical", "Hybrid", "Quantum"):
        sub = arch[arch["Family"] == family]
        if sub.empty:
            continue
        avg_out = sub["outgoing_score"].mean()
        avg_in = sub["incoming_score"].mean()
        lines.append(
            f"- **{family}**: mean outgoing={avg_out:.4f}, mean incoming={avg_in:.4f} "
            f"({len(sub)} architectures)."
        )

    outliers = detect_outliers(arch)
    outlier_rows = outliers[outliers["is_outlier"]] if "is_outlier" in outliers.columns else pd.DataFrame()
    lines.extend(["", "## Outliers (architecture-level)", ""])
    if outlier_rows.empty:
        lines.append("- No architecture points flagged as outliers.")
    else:
        for _, r in outlier_rows.iterrows():
            lines.append(
                f"- **{r['Architecture']}**: distance={r['distance_from_mean']:.4f}, "
                f"quadrant={r['quadrant']} — {r.get('quadrant_label', '')}"
            )

    return "\n".join(lines) + "\n"


def _outliers_markdown(model_df: pd.DataFrame, arch_df: pd.DataFrame) -> str:
    lines = ["# Outliers and quadrant assignments", ""]
    for name, df in [("Architecture", arch_df), ("Configuration", model_df)]:
        if df.empty:
            continue
        tagged = detect_outliers(add_quadrants(df))
        lines.append(f"## {name}-level")
        lines.append("")
        for _, r in tagged.iterrows():
            label = r.get("Architecture", r.get("model", "?"))
            flag = " **outlier**" if r.get("is_outlier") else ""
            lines.append(
                f"- {label}{flag}: quadrant={r['quadrant']}, "
                f"out={r['outgoing_score']:.4f}, in={r['incoming_score']:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def run_transfer_scatter_figures(
    pair_df: pd.DataFrame,
    out_dir: Path,
    *,
    project_root: Path | None = None,
    plot_options: TransferScatterPlotOptions | None = None,
    filter_modes: Sequence[TransferFilterMode] = (
        "inter_model",
        "inter_architecture",
        "inter_family",
        "attack_lens_classical",
        "attack_lens_hybrid",
        "attack_lens_quantum",
    ),
    generate_architecture: bool = True,
    generate_variant: bool = True,
) -> dict[str, Path]:
    """Generate transfer scatter PNGs only (no CSV tables or interpretation files)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or _resolve_project_root(out_dir, None)
    opts = plot_options or TransferScatterPlotOptions()
    resolve_transfer_metric(pair_df, opts.metric)
    metric_slug = metric_file_slug(opts.metric)
    outputs: dict[str, Path] = {}

    for mode in filter_modes:
        model_df = compute_model_transfer_profile(pair_df, mode, metric=opts.metric)
        arch_df = aggregate_architecture_profile(model_df)
        suffix = f"{mode}__{metric_slug}"

        if generate_architecture:
            arch_path = out_dir / f"architecture_{suffix}.png"
            plot_architecture_transfer_scatter(
                arch_df,
                filter_mode=mode,
                out_path=arch_path,
                plot_options=opts,
            )
            outputs[f"architecture_{suffix}"] = arch_path

        if generate_variant:
            variant_path = out_dir / f"variant_{suffix}.png"
            plot_variant_transfer_scatter(
                model_df,
                filter_mode=mode,
                out_path=variant_path,
                project_root=root,
                plot_options=opts,
            )
            outputs[f"variant_{suffix}"] = variant_path

    log.info("Transfer scatter figures written to %s", out_dir)
    return outputs


def run_transfer_scatter_report(
    pair_df: pd.DataFrame,
    out_dir: Path,
    *,
    project_root: Path | None = None,
    plot_options: TransferScatterPlotOptions | None = None,
    filter_modes: Sequence[TransferFilterMode] = (
        "inter_model",
        "inter_architecture",
        "inter_family",
        "attack_lens_classical",
        "attack_lens_hybrid",
        "attack_lens_quantum",
    ),
) -> dict[str, Path]:
    """Generate tables, figures, and interpretation for all filter modes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = project_root or _resolve_project_root(out_dir, None)
    opts = plot_options or TransferScatterPlotOptions()
    outputs: dict[str, Path] = {}

    outputs.update(
        run_transfer_scatter_figures(
            pair_df,
            out_dir,
            project_root=root,
            plot_options=opts,
            filter_modes=filter_modes,
        )
    )

    all_model: list[pd.DataFrame] = []
    all_arch: list[pd.DataFrame] = []

    metric = opts.metric
    for mode in filter_modes:
        model_df = compute_model_transfer_profile(pair_df, mode, metric=metric)
        arch_df = aggregate_architecture_profile(model_df)
        all_model.append(model_df.assign(filter_mode=mode))
        all_arch.append(arch_df.assign(filter_mode=mode))

        suffix = mode
        model_path = out_dir / f"model_profile_{suffix}.csv"
        arch_path = out_dir / f"architecture_profile_{suffix}.csv"
        model_df.to_csv(model_path, index=False)
        arch_df.to_csv(arch_path, index=False)
        outputs[f"model_profile_{suffix}"] = model_path
        outputs[f"architecture_profile_{suffix}"] = arch_path

    arch_inter = aggregate_architecture_profile(
        compute_model_transfer_profile(pair_df, "inter_model", metric=metric)
    )
    summary = build_transfer_summary_table(arch_inter)
    summary_path = out_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    outputs["summary_table"] = summary_path

    model_inter = compute_model_transfer_profile(pair_df, "inter_model", metric=metric)
    interp = generate_interpretation_notes(arch_inter, model_inter, metric=metric)
    interp_path = out_dir / "interpretation.md"
    interp_path.write_text(interp, encoding="utf-8")
    outputs["interpretation"] = interp_path

    outliers_path = out_dir / "outliers.md"
    outliers_path.write_text(
        _outliers_markdown(model_inter, arch_inter),
        encoding="utf-8",
    )
    outputs["outliers"] = outliers_path

    pd.concat(all_model, ignore_index=True).to_csv(
        out_dir / "model_profile_all_modes.csv", index=False
    )
    pd.concat(all_arch, ignore_index=True).to_csv(
        out_dir / "architecture_profile_all_modes.csv", index=False
    )

    log.info("Transfer scatter report written to %s", out_dir)
    return outputs
