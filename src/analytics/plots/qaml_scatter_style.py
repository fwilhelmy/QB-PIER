"""
QAML-style scatter figures (white-box reference aesthetics).

Shared encoding: ansatz bucket markers, family outline, circuit-depth fill,
parameter-scaled point size, seaborn whitegrid theme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

FIGURE_DPI = 160
FIGURE_SIZE = (12.6, 8.0)
DEPTH_CMAP_NAME = "YlOrRd"
MISSING_DEPTH_COLOR = "#C5CAD3"

ANSATZ_LEGEND_LABELS = {
    "No quantum": "None (Classical)",
    "Basic ansatz": "Basic (Ring Entanglement)",
    "Random ansatz": "Random (Random Structure)",
    "Strong ansatz": "Strong (Expressive Entanglement)",
}

FAMILY_EDGE_COLORS = {
    "Classical": "#2E4780",
    "Hybrid": "#804126",
    "Quantum": "#386411",
}

FAMILY_LINEWIDTHS = {
    "Classical": 1.35,
    "Hybrid": 1.85,
    "Quantum": 1.35,
}


def make_unique(names: Sequence[object]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = str(name).strip()
        if not base or base.lower() == "nan":
            base = "blank"
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base} {counts[base]}")
    return out


def read_grouped_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    columns = make_unique(raw.iloc[0].tolist())
    df = raw.iloc[1:].copy()
    df.columns = columns
    return df.dropna(how="all").reset_index(drop=True)


def num_to_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", "")
    if not text or text == "-":
        return float("nan")
    return float(text)


def parse_config(config: object) -> dict[str, str]:
    text = "" if pd.isna(config) else str(config)
    out: dict[str, str] = {}
    for part in text.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def architecture_identity(row: pd.Series) -> str:
    arch = str(row["Architecture"])
    if arch.upper() == "VQM":
        cfg = parse_config(row.get("Variant Configuration (knobs)"))
        ansatz = cfg.get("ansatz", "unknown")
        depth = cfg.get("qdepth")
        return f"VQM {ansatz}" if depth is None else f"VQM {ansatz} q{depth}"
    return arch


def architecture_label(row: pd.Series) -> str:
    arch = str(row["Architecture"])
    if arch.upper() == "VQM":
        return "VQM"
    if arch.upper() == "HVQL":
        config = (
            ""
            if pd.isna(row.get("Variant Configuration (knobs)"))
            else str(row.get("Variant Configuration (knobs)")).strip().lower()
        )
        if config == "dropout":
            return "HVQL-Dropout"
        if config == "residuals":
            return "HVQL-Residual"
    return arch


def ansatz_bucket(row: pd.Series) -> str:
    family = str(row.get("Group / Family", row.get("family", "")))
    if family == "Classical":
        return "No quantum"
    arch = str(row["Architecture"]).upper()
    if arch == "HQUANV":
        return "Basic ansatz"
    if arch == "HQE":
        return "Strong ansatz"
    if arch == "HVQL":
        config = (
            ""
            if pd.isna(row.get("Variant Configuration (knobs)"))
            else str(row.get("Variant Configuration (knobs)")).strip().lower()
        )
        if config in {"dropout", "residuals"}:
            return "Strong ansatz"
    cfg = parse_config(row.get("Variant Configuration (knobs)"))
    ansatz = cfg.get("ansatz", "").lower()
    if ansatz == "basic":
        return "Basic ansatz"
    if ansatz == "random":
        return "Random ansatz"
    if ansatz == "strong":
        return "Strong ansatz"
    return "No quantum"


def marker_for_bucket(bucket: str) -> str:
    return {
        "No quantum": "o",
        "Basic ansatz": "s",
        "Random ansatz": "D",
        "Strong ansatz": "^",
    }.get(bucket, "o")


def _results_paths(project_root: Path) -> list[Path]:
    names = ["Results - Training.csv", "Results - QuantumCircuits.csv"]
    return [project_root / n for n in names if (project_root / n).exists()]


def build_params_lookup(project_root: Path) -> tuple[pd.DataFrame, dict[tuple[str, str, str], float]]:
    frames: list[pd.DataFrame] = []
    for path in _results_paths(project_root):
        frame = read_grouped_csv(path)
        if "Params Total" in frame.columns:
            frame["Params Total"] = frame["Params Total"].map(num_to_float)
            frames.append(
                frame[
                    [
                        "Group / Family",
                        "Architecture",
                        "Variant Configuration (knobs)",
                        "Params Total",
                    ]
                ]
            )
    if not frames:
        return pd.DataFrame(), {}
    params = pd.concat(frames, ignore_index=True).dropna(subset=["Params Total"])
    exact = {
        (
            str(row["Group / Family"]),
            str(row["Architecture"]),
            ""
            if pd.isna(row["Variant Configuration (knobs)"])
            else str(row["Variant Configuration (knobs)"]),
        ): float(row["Params Total"])
        for _, row in params.iterrows()
    }
    return params, exact


def attach_params(
    df: pd.DataFrame,
    project_root: Path | None = None,
    *,
    params_dump_path: Path | None = None,
) -> pd.DataFrame:
    """Attach ``Params Total``; prefers ``analytics/attacks/dump_params_size.json``."""
    root = project_root or Path.cwd()
    params, exact = build_params_lookup(root)
    dump: dict[str, float] = {}
    lookup_fn = None
    try:
        from analytics.attacks.params_size import load_params_size_dump, lookup_params_total

        dump = load_params_size_dump(
            str(params_dump_path) if params_dump_path is not None else None
        )
        lookup_fn = lookup_params_total
    except ImportError:
        pass

    values: list[float] = []
    for _, row in df.iterrows():
        if dump and lookup_fn is not None:
            val = lookup_fn(
                None if pd.isna(row.get("model")) else str(row.get("model")),
                dump,
                family=str(row.get("Group / Family", "")),
                architecture=str(row.get("Architecture", "")),
                variant=row.get("Variant Configuration (knobs)"),
            )
            if np.isfinite(val):
                values.append(val)
                continue

        config = (
            ""
            if pd.isna(row.get("Variant Configuration (knobs)"))
            else str(row.get("Variant Configuration (knobs)"))
        )
        key = (str(row["Group / Family"]), str(row["Architecture"]), config)
        if key in exact:
            values.append(exact[key])
            continue
        cfg = parse_config(config)
        candidates = params.copy()
        if not candidates.empty:
            candidates = candidates[
                (candidates["Group / Family"].astype(str) == str(row["Group / Family"]))
                & (candidates["Architecture"].astype(str) == str(row["Architecture"]))
            ]
            if cfg:
                for field in ["ansatz", "qdepth"]:
                    if field in cfg:
                        candidates = candidates[
                            candidates["Variant Configuration (knobs)"]
                            .astype(str)
                            .str.contains(f"{field}={cfg[field]}", regex=False, na=False)
                        ]
        if len(candidates):
            values.append(float(candidates["Params Total"].median()))
        else:
            values.append(float("nan"))
    out = df.copy()
    out["Params Total"] = values
    return out


def build_depth_lookup(project_root: Path) -> pd.DataFrame:
    path = project_root / "Results - QuantumCircuits.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = read_grouped_csv(path)
    if "Layers" in frame.columns:
        frame["Circuit Depth Value"] = frame["Layers"].map(num_to_float)
    elif "Circuit Depth" in frame.columns:
        frame["Circuit Depth Value"] = frame["Circuit Depth"].map(num_to_float)
    else:
        frame["Circuit Depth Value"] = np.nan
    return frame[
        ["Group / Family", "Architecture", "Variant Configuration (knobs)", "Circuit Depth Value"]
    ].dropna(subset=["Circuit Depth Value"])


def _depth_from_row_rules(row: pd.Series, circuits: pd.DataFrame) -> float:
    arch = str(row["Architecture"]).upper()
    if arch == "HQUANV":
        return 2.0
    if arch == "HQE":
        return 50.0
    if arch == "HVQL":
        config = (
            ""
            if pd.isna(row.get("Variant Configuration (knobs)"))
            else str(row.get("Variant Configuration (knobs)")).strip().lower()
        )
        if config in {"dropout", "residuals"}:
            return 50.0
    cfg = parse_config(row.get("Variant Configuration (knobs)"))
    if "qdepth" in cfg:
        return float(cfg["qdepth"])
    if circuits.empty or str(row["Group / Family"]) == "Classical":
        return float("nan")
    candidates = circuits[
        (circuits["Group / Family"].astype(str) == str(row["Group / Family"]))
        & (circuits["Architecture"].astype(str) == str(row["Architecture"]))
    ].copy()
    config = (
        ""
        if pd.isna(row.get("Variant Configuration (knobs)"))
        else str(row.get("Variant Configuration (knobs)"))
    )
    if config and len(candidates):
        exact = candidates[candidates["Variant Configuration (knobs)"].astype(str) == config]
        if len(exact):
            candidates = exact
    return float(candidates["Circuit Depth Value"].median()) if len(candidates) else float("nan")


def attach_depth(df: pd.DataFrame, project_root: Path | None = None) -> pd.DataFrame:
    root = project_root or Path.cwd()
    circuits = build_depth_lookup(root)
    values = [_depth_from_row_rules(row, circuits) for _, row in df.iterrows()]
    out = df.copy()
    out["Circuit Depth Value"] = values
    return out


def depth_color(value: float, norm: Normalize, cmap) -> str:
    if pd.isna(value):
        return MISSING_DEPTH_COLOR
    return cmap(norm(value))


def param_to_size(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    fallback = np.nanmedian(values) if np.isfinite(values).any() else 1000.0
    values = np.where(np.isfinite(values), values, fallback)
    logs = np.log10(np.maximum(values, 1.0))
    lo, hi = np.nanmin(logs), np.nanmax(logs)
    if hi == lo:
        return np.full_like(logs, 90.0)
    return 36.0 + ((logs - lo) / (hi - lo)) * 150.0


def finalize_enriched_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply visual columns after coordinate aggregation (e.g. architecture-level means)."""
    out = df.copy()
    out["Architecture Identity"] = out.apply(architecture_identity, axis=1)
    out["Architecture Label"] = out.apply(architecture_label, axis=1)
    out["Ansatz Bucket"] = out.apply(ansatz_bucket, axis=1)
    out["Marker"] = out["Ansatz Bucket"].map(marker_for_bucket)
    out["Point Size"] = param_to_size(out["Params Total"].to_numpy())
    if "outgoing_transfer_asr" in out.columns:
        out["_plot_x"] = pd.to_numeric(out["outgoing_transfer_asr"], errors="coerce")
    if "incoming_transfer_asr" in out.columns:
        out["_plot_y"] = pd.to_numeric(out["incoming_transfer_asr"], errors="coerce")
    return out


def enrich_plot_schema(
    df: pd.DataFrame,
    *,
    project_root: Path | None = None,
    x_col: str = "outgoing_transfer_asr",
    y_col: str = "incoming_transfer_asr",
) -> pd.DataFrame:
    """Add white-box-style visual columns; expects family/architecture/variant or Group/Arch/knobs."""
    work = df.copy()
    if "Group / Family" not in work.columns:
        work["Group / Family"] = work.get("Family", work.get("family"))
    if "Architecture" not in work.columns:
        work["Architecture"] = work.get("architecture")
    if "Variant Configuration (knobs)" not in work.columns:
        work["Variant Configuration (knobs)"] = work.get("variant", "")

    root = project_root
    if root is None and "project_root" in work.columns:
        root = Path(str(work["project_root"].iloc[0]))
    work = attach_params(work, root)
    work = attach_depth(work, root)
    work["Architecture Identity"] = work.apply(architecture_identity, axis=1)
    work["Architecture Label"] = work.apply(architecture_label, axis=1)
    work["Ansatz Bucket"] = work.apply(ansatz_bucket, axis=1)
    work["Marker"] = work["Ansatz Bucket"].map(marker_for_bucket)
    work["Point Size"] = param_to_size(work["Params Total"].to_numpy())
    work["_plot_x"] = pd.to_numeric(work[x_col], errors="coerce")
    work["_plot_y"] = pd.to_numeric(work[y_col], errors="coerce")
    return work.dropna(subset=["_plot_x", "_plot_y"])


def draw_quadrant_region_annotations(
    ax: plt.Axes,
    x_mean: float,
    y_mean: float,
    *,
    labels: dict[str, str] | None = None,
    fontsize: float = 9.0,
    color: str = "#6B7280",
    alpha: float = 0.72,
) -> None:
    """Place one label per quadrant at the region centre."""
    label_map = labels or {
        "TR": "High Attack Potential\nHigh Robustness Potential",
        "BR": "High Attack Potential\nLow Robustness Potential",
        "TL": "Low Attack Potential\nHigh Robustness Potential",
        "BL": "Low Attack Potential\nLow Robustness Potential",
    }
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    centres = {
        "TR": ((x_mean + xhi) * 0.5, (y_mean + yhi) * 0.5),
        "BR": ((x_mean + xhi) * 0.5, (ylo + y_mean) * 0.5),
        "TL": ((xlo + x_mean) * 0.5, (y_mean + yhi) * 0.5),
        "BL": ((xlo + x_mean) * 0.5, (ylo + y_mean) * 0.5),
    }
    for key, (cx, cy) in centres.items():
        text = label_map.get(key)
        if not text:
            continue
        ax.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            alpha=alpha,
            zorder=2,
            linespacing=1.15,
        )


def render_scatter_chart(
    df: pd.DataFrame,
    *,
    out_path: Path,
    x_col: str = "_plot_x",
    y_col: str = "_plot_y",
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    show_labels: bool = True,
    quadrant_means: tuple[float, float] | None = None,
    show_quadrant_annotations: bool = False,
    quadrant_annotation_labels: dict[str, str] | None = None,
    clip_axes: bool = True,
    xlim: tuple[float, float] = (0.4, 1.0),
    ylim: tuple[float, float] = (0.4, 1.0),
) -> plt.Figure:
    """Render QAML-style scatter; *df* must already be enriched via ``enrich_plot_schema``."""
    if df.empty:
        raise ValueError("Cannot render scatter chart from empty dataframe.")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.patch.set_facecolor("#FCFCFD")
    ax.set_facecolor("#FFFFFF")

    depth_values = df["Circuit Depth Value"].dropna()
    if len(depth_values):
        norm = Normalize(vmin=float(depth_values.min()), vmax=float(depth_values.max()))
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(DEPTH_CMAP_NAME)

    for _, row in df.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            s=float(row["Point Size"]),
            marker=row["Marker"],
            facecolor=depth_color(row["Circuit Depth Value"], norm, cmap),
            edgecolor=FAMILY_EDGE_COLORS.get(str(row["Group / Family"]), "#464C55"),
            alpha=0.92,
            linewidth=FAMILY_LINEWIDTHS.get(str(row["Group / Family"]), 1.2),
            zorder=3,
        )

    if clip_axes:
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
    else:
        x_max = float(df[x_col].max())
        y_max = float(df[y_col].max())
        ax.set_xlim(0, min(1.0, max(0.15, x_max + 0.04)))
        ax.set_ylim(0, min(1.0, max(0.15, y_max + 0.04)))

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=14, pad=10)

    ax.grid(axis="y", color="#E0E4EF", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="#F0F2F6", linestyle="-", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D7DBE7")

    if quadrant_means is not None:
        x_mean, y_mean = quadrant_means
        ax.axvline(x_mean, color="#9AA3B8", linestyle="--", linewidth=0.9, zorder=1)
        ax.axhline(y_mean, color="#9AA3B8", linestyle="--", linewidth=0.9, zorder=1)
        if show_quadrant_annotations:
            draw_quadrant_region_annotations(
                ax,
                x_mean,
                y_mean,
                labels=quadrant_annotation_labels,
            )

    if show_labels:
        offsets = [(5, 5), (5, -8), (-18, 5), (-18, -8), (8, 12), (-24, 12)]
        for i, (_, row) in enumerate(df.iterrows()):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                str(row["Architecture Label"]),
                (row[x_col], row[y_col]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.2,
                color="#1F2430",
                alpha=0.78,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "fc": "#FCFCFD",
                    "ec": "none",
                    "alpha": 0.58,
                },
            )

    present_buckets = set(df["Ansatz Bucket"].astype(str))
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_for_bucket(label),
            color="none",
            markerfacecolor="#D8DCE6",
            markeredgecolor="#1F2430",
            markersize=8,
            label=ANSATZ_LEGEND_LABELS[label],
        )
        for label in ANSATZ_LEGEND_LABELS
        if label in present_buckets
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#FFFFFF",
            markeredgecolor=color,
            markeredgewidth=1.8,
            markersize=8,
            label=family,
        )
        for family, color in FAMILY_EDGE_COLORS.items()
        if family in set(df["Group / Family"].astype(str))
    ]
    legend1 = ax.legend(
        handles=shape_handles,
        loc="upper left",
        frameon=True,
        title="Ansatz Structure",
        fontsize=8,
        title_fontsize=10,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=family_handles,
        loc="lower right",
        frameon=True,
        title="Family outline",
        fontsize=8,
        title_fontsize=10,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("Circuit depth")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=FIGURE_DPI)
    plt.close(fig)
    return fig
