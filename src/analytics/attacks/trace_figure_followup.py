"""Final thesis figure set derived only from saved mechanistic trace artifacts.

This module is library code.  It does not load a model, run inference, or create
an attack.  Clean margins are recovered from the already-saved clean-logit
baselines and joined to the saved per-sample layer traces before aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import shutil
import sys
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"
FIGURE_DIR = OUTPUT_ROOT / "followup_figures"
DATA_DIR = OUTPUT_ROOT / "followup_figure_data"
METHOD_DIR = OUTPUT_ROOT / "followup_methods"
CAPTION_DIR = OUTPUT_ROOT / "followup_captions"
SCRIPT_DIR = OUTPUT_ROOT / "followup_scripts"
NOTEBOOK_DIR = OUTPUT_ROOT / "followup_notebooks"
TRACE_ROOT = PROJECT_ROOT / "outputs" / "mechanistic_visualizations"
LEGACY_ROOT = PROJECT_ROOT / "outputs" / "mechanistic_propagation"
ATTACK_ROOT = PROJECT_ROOT / "logs" / "ATTACKS" / "attack_experiment_canonical"

EPSILON = 0.10
MARGIN_TAU = 1e-3
MARGIN_THRESHOLD = 1e-3
EFFICIENCY_TAU = 1e-3
ALIGNMENT_THRESHOLD = 1e-4
DETERMINISTIC_SEED = 20260727


COLORS: Mapping[str, str] = {
    "cnn": "#0072B2",
    "resnet18": "#332288",
    "mlp_large": "#D55E00",
    "mlp_small": "#E69F00",
    "mlpmixer": "#CC79A7",
    "hquanv": "#117733",
    "hvql": "#0072B2",
    "hpvql5": "#009E73",
    "hpvql10": "#56B4E9",
    "hvqm": "#AA4499",
    "hqe": "#999933",
    "reshvql": "#D55E00",
    "drophvql": "#CC79A7",
}

MARKERS: Mapping[str, str] = {
    "cnn": "h", "resnet18": "s", "mlp_large": "v", "mlp_small": "^",
    "mlpmixer": "D", "hquanv": "P", "hvql": "o", "hpvql5": "s",
    "hpvql10": "^", "hvqm": "D", "hqe": "P", "reshvql": "X",
    "drophvql": "v",
}

LINESTYLES: Mapping[str, str] = {
    "cnn": "--", "resnet18": "--", "mlp_large": ":", "mlp_small": "-.",
    "mlpmixer": "-", "hquanv": "--", "hvql": "-", "hpvql5": "--",
    "hpvql10": ":", "hvqm": "-.", "hqe": ":", "reshvql": "--",
    "drophvql": "-.",
}

DISPLAY_LABELS: Mapping[str, str] = {
    "cnn": "CNN",
    "resnet18": "ResNet18",
    "mlp_large": "MLP [1024,512,64,64]",
    "mlp_small": "MLP [128,64]",
    "mlpmixer": "MLPMixer",
    "hquanv": "HQuanv",
    "hvql": "HVQL",
    "hpvql5": "HPVQL (5 blocs)",
    "hpvql10": "HPVQL (10 blocs)",
    "hvqm": "HVQM",
    "hqe": "HQE",
    "reshvql": "HVQL résiduel",
    "drophvql": "HVQL dropout",
}

BASELINE_FOLDERS: Mapping[str, str] = {
    "cnn": "Classical__CNN__conv_channels=[32,64],_fc_sizes=[128]",
    "resnet18": "Classical__resnet18",
    "mlp_large": "Classical__MLP__hidden_sizes=[1024,512,64,64]",
    "mlp_small": "Classical__mlp__hidden_sizes=[128,64]",
    "mlpmixer": "Classical__mlpmixer",
    "hquanv": "Hybrid__hquanv",
    "hvql": "Hybrid__HVQL__ansatz=strong,_qdepth=50",
    "hpvql5": "Hybrid__HPVQL__qblocks=5,_qdepth=50,_ansatz=strong",
    "hpvql10": "Hybrid__HPVQL__qblocks=10,_qdepth=50,_ansatz=strong",
    "hvqm": "Hybrid__HVQM__ansatz=strong,_qdepth=50",
    "hqe": "Hybrid__hqe",
    "reshvql": "Hybrid__hvql__residuals",
    "drophvql": "Hybrid__hvql__dropout",
}


@dataclass(frozen=True)
class ModelTrace:
    key: str
    source: str
    layers: tuple[str, ...]
    evidence: str


MODEL_TRACES: Mapping[str, ModelTrace] = {
    "hvql": ModelTrace("hvql", "layerwise", (
        "input", "conv1", "pool1", "conv2", "encoder", "interface_dense",
        "prequantum", "measurement", "head", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    "cnn": ModelTrace("cnn", "layerwise", (
        "input", "conv1", "pool1", "conv2", "encoder", "dense",
        "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    "mlp_large": ModelTrace("mlp_large", "layerwise", (
        "input", "dense1", "dense2", "dense3", "predecision", "head", "logits",
    ), "500 exemples par graine; graines 0 et 42; modèle non attaqué dans le balayage historique"),
    "mlpmixer": ModelTrace("mlpmixer", "layerwise", (
        "input", "patch_projection", "patch_tokens", "mixer1", "mixer2",
        "mixer3", "encoder", "predecision", "head", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    "reshvql": ModelTrace("reshvql", "legacy", (
        "raw input", "conv 1", "pool 1", "conv 2", "encoder output",
        "interface dense", "pre-quantum latent", "quantum branch",
        "classical bypass", "residual merge", "final logits",
    ), "diagnostic commun-correct de 19 exemples; graine 0"),
}

LEGACY_LABELS: Mapping[str, str] = {
    "raw input": "Entrée", "conv 1": "Conv. 1", "pool 1": "Pool 1",
    "conv 2": "Conv. 2", "encoder output": "Encodeur",
    "interface dense": "Interface dense", "pre-quantum latent": "Latent préquantique",
    "measured quantum vector": "Mesures quantiques", "measured quantum logits": "Mesures quantiques",
    "classical head": "Lecture classique", "final logits": "Logits",
    "quantum branch": "Branche quantique", "classical bypass": "Contournement classique",
    "residual merge": "Fusion résiduelle", "pre-dropout latent": "Latent pré-dropout",
    "inference dropout": "Dropout (inférence)", "dense hidden": "Couche dense",
    "dense ReLU": "Activation dense", "dense 784->128": "Dense 784→128",
    "dense 128->64": "Dense 128→64", "patch projection": "Projection patchs",
    "patch tokens": "Jetons patchs", "mixer block 1": "Mixer 1",
    "mixer block 2": "Mixer 2", "mixer block 3": "Mixer 3",
    "mixer block 4": "Mixer 4", "token normalization": "Normalisation",
    "stem convolution": "Tige conv.", "residual stage 2": "Résiduel 2",
    "global average pool": "Moyenne globale", "linear head": "Tête linéaire",
    "quanvolution map": "Quanvolution", "flattened quanvolution": "Carte quanvolution",
    "dense 784->4096": "Dense 784→4096", "dense 4096->2048": "Dense 4096→2048",
    "encoder latent 1024": "Latent 1024",
}


def source_paths() -> dict[str, Path]:
    return {
        "layerwise": TRACE_ROOT / "trace_layerwise_samples.csv",
        "interface": TRACE_ROOT / "trace_interface_samples.csv",
        "segmentation": TRACE_ROOT / "trace_segmentation_samples.csv",
        "residual": TRACE_ROOT / "trace_residual_samples.csv",
        "legacy": LEGACY_ROOT / "layerwise_trace_samples.csv",
        "checkpoint_alignment": TRACE_ROOT / "checkpoint_alignment.csv",
        "architecture_audit": TRACE_ROOT / "architecture_label_audit.csv",
    }


def discover_sources() -> pd.DataFrame:
    rows = []
    for key, path in source_paths().items():
        rows.append({
            "source_key": key,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def load_artifacts() -> dict[str, pd.DataFrame]:
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Artefacts absents : " + ", ".join(missing))
    return {key: pd.read_csv(path) for key, path in paths.items()}


def _clean_margin(model_key: str, seed: int, indices: np.ndarray) -> np.ndarray:
    labels_path = ATTACK_ROOT / f"seed_{seed}" / "labels.npy"
    logits_path = ATTACK_ROOT / f"seed_{seed}" / "baselines" / BASELINE_FOLDERS[model_key] / "logits_clean.npy"
    if not labels_path.exists() or not logits_path.exists():
        raise FileNotFoundError(f"Logits propres absents pour {model_key}, graine {seed}")
    labels = np.load(labels_path)
    logits = np.load(logits_path).astype(np.float64)
    y = labels[indices].astype(int)
    selected = logits[indices]
    competitors = selected.copy()
    competitors[np.arange(len(indices)), y] = -np.inf
    return selected[np.arange(len(indices)), y] - competitors.max(axis=1)


def _enrich_saved_trace(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    work = frame[np.isclose(frame["epsilon"], EPSILON)].copy()
    if "condition" in work:
        condition = "auto-FGSM" if "auto-FGSM" in set(work["condition"]) else "self"
        work = work[work["condition"] == condition].copy()
    work["clean_margin"] = np.nan
    for (model_key, seed), group in work.groupby(["model_key", "seed"], sort=False):
        indices = group["dataset_index"].to_numpy(int)
        work.loc[group.index, "clean_margin"] = _clean_margin(str(model_key), int(seed), indices)
    work["margin_normalization_covered"] = work["clean_margin"] >= MARGIN_THRESHOLD
    work["margin_normalized_harm"] = np.where(
        work["margin_normalization_covered"],
        work["projected_margin_harm"] / (work["clean_margin"] + MARGIN_TAU),
        np.nan,
    )
    if "gain_rms" in work:
        work["harm_efficiency"] = work["margin_normalized_harm"] / (work["gain_rms"] + EFFICIENCY_TAU)
    work["source_artifact"] = source_name
    return work


def _enrich_residual(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[np.isclose(frame["epsilon"], EPSILON)].copy()
    work["clean_margin"] = np.nan
    for (model_key, seed), group in work.groupby(["model_key", "seed"], sort=False):
        indices = group["dataset_index"].to_numpy(int)
        work.loc[group.index, "clean_margin"] = _clean_margin(str(model_key), int(seed), indices)
    work["margin_normalization_covered"] = work["clean_margin"] >= MARGIN_THRESHOLD
    work["margin_normalized_harm"] = np.where(
        work["margin_normalization_covered"],
        work["projected_margin_harm"] / (work["clean_margin"] + MARGIN_TAU),
        np.nan,
    )
    work["source_artifact"] = "trace_residual_samples.csv"
    return work


def _enrich_legacy(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[np.isclose(frame["epsilon"], EPSILON) & (frame["condition"] == "self")].copy()
    work["seed"] = 0
    work["dataset_index"] = work["sample"].astype(int)
    work["layer_fr"] = work["layer"].map(LEGACY_LABELS).fillna(work["layer"])
    work["projected_margin_harm"] = work["predicted_margin_drop"]
    work["alignment_harmful"] = -work["margin_gradient_alignment"]
    work["margin_normalization_covered"] = work["clean_margin"] >= MARGIN_THRESHOLD
    work["margin_normalized_harm"] = np.where(
        work["margin_normalization_covered"],
        work["projected_margin_harm"] / (work["clean_margin"] + MARGIN_TAU),
        np.nan,
    )
    work["harm_efficiency"] = work["margin_normalized_harm"] / (work["gain_rms"] + EFFICIENCY_TAU)
    denominator = work["delta_l2"] * work["alignment_harmful"].abs()
    work["gradient_sensitivity"] = np.where(
        (work["alignment_harmful"].abs() >= ALIGNMENT_THRESHOLD) & (work["delta_l2"] > 0),
        work["projected_margin_harm"].abs() / denominator.replace(0, np.nan),
        np.nan,
    )
    work["relative_displacement"] = work["relative_l2"]
    work["source_artifact"] = "layerwise_trace_samples.csv"
    return work


def prepared_data(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "layerwise": _enrich_saved_trace(artifacts["layerwise"], "trace_layerwise_samples.csv"),
        "interface": _enrich_saved_trace(artifacts["interface"], "trace_interface_samples.csv"),
        "segmentation": _enrich_saved_trace(artifacts["segmentation"], "trace_segmentation_samples.csv"),
        "residual": _enrich_residual(artifacts["residual"]),
        "legacy": _enrich_legacy(artifacts["legacy"]),
    }


def metric_definition_audit() -> pd.DataFrame:
    rows = [
        ("Déplacement RMS brut", "delta_rms", "sqrt(mean_j(Δa_j²))", "par dimension", "médiane échantillon puis médiane des graines", "taille du déplacement dans la représentation", "Les unités changent avec l’échelle des activations."),
        ("Gain RMS normalisé à l’entrée", "gain_rms", "RMS(Δa_l) / RMS(δ)", "perturbation d’entrée du même échantillon", "calcul échantillon, puis médiane/IQR", "progression de la perturbation entre dimensions différentes", "Ne mesure pas la pertinence décisionnelle."),
        ("Gain RMS bloc-à-bloc", "dérivé: delta_rms_l / delta_rms_{l-1}", "RMS(Δa_l) / RMS(Δa_{l-1})", "bloc précédent explicitement sélectionné", "calcul échantillon avant agrégation", "amplification ou atténuation locale", "Dépend du choix de deux blocs homologues; non utilisé comme trajectoire principale."),
        ("Dommage de marge brut", "projected_margin_harm", "-∇_{a_l}m_f(x)^T Δa_l", "aucune", "calcul échantillon, puis médiane/IQR", "réduction locale projetée de la marge si positif", "Approximation locale; conserve le signe."),
        ("Dommage normalisé par la marge propre", "dérivé: projected_margin_harm / (clean_margin + 1e-3)", "H_l / (m_f(x)+τ)", "marge propre par échantillon; τ=1e-3", "calcul échantillon avant agrégation", "fraction locale de la marge propre consommée", "Exclut m_f(x)<1e-3; ne pas confondre avec normalized_margin_harm sauvegardé, normalisé par |H_entrée|."),
        ("Déplacement relatif", "relative_l2", "||Δa_l||₂ / (||a_l||₂+τ)", "activation propre de la même couche", "calcul échantillon, puis médiane/IQR", "déplacement relativement à l’amplitude propre", "Disponible seulement dans le diagnostic historique de 19 exemples; absent pour le grand MLP."),
        ("Sensibilité du gradient", "gradient_sensitivity", "||∇_{a_l}m_f(x)||₂", "aucune", "calcul échantillon, puis médiane/IQR", "sensibilité aval de la marge", "Pour les traces historiques, récupérée par |H|/(||Δa||₂|C|) et exclue si |C|<1e-4."),
        ("Alignement nuisible", "alignment_harmful", "-cos(∇_{a_l}m_f(x), Δa_l)", "normes L2 des deux vecteurs", "calcul échantillon, puis médiane/IQR", "compatibilité directionnelle avec une réduction de marge", "Ne mesure pas la taille du déplacement."),
        ("Efficacité du dommage", "dérivé: margin_normalized_harm / (gain_rms + 1e-3)", "H̃_l/(G_l+τ_E)", "gain RMS; τ_E=1e-3", "calcul échantillon avant agrégation", "dommage normalisé par unité de gain RMS", "Peut devenir instable lorsque G_l est très faible; utilisé seulement avec couverture et IQR explicites."),
    ]
    columns = ["metric_name", "source_column", "formula", "normalization", "aggregation", "interpretation", "caveat"]
    return pd.DataFrame(rows, columns=columns)


def _model_frame(data: Mapping[str, pd.DataFrame], spec: ModelTrace) -> pd.DataFrame:
    work = data[spec.source]
    raw_key = "reshvql" if spec.key == "reshvql" else spec.key
    work = work[work["model_key"] == raw_key].copy()
    order = {layer: position for position, layer in enumerate(spec.layers)}
    work = work[work["layer"].isin(order)].copy()
    work["stage_order"] = work["layer"].map(order).astype(int)
    work["stage_label"] = work["layer_fr"]
    work["model"] = DISPLAY_LABELS[spec.key]
    return work.sort_values(["stage_order", "seed", "dataset_index"])


def _seed_name(seed: object) -> str:
    try:
        return f"seed_{int(seed)}_median"
    except (TypeError, ValueError):
        return f"seed_{seed}_median"


def summarize(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    work = frame.dropna(subset=[metric]).copy()
    if work.empty:
        return pd.DataFrame()
    group_cols = ["model_key", "model", "stage_order", "stage_label", "layer"]
    pooled = work.groupby(group_cols, as_index=False)[metric].agg(
        q25=lambda x: x.quantile(.25), pooled_median="median", q75=lambda x: x.quantile(.75),
        n_rows="size",
    )
    sample_counts = work.groupby(group_cols, as_index=False)["dataset_index"].nunique().rename(
        columns={"dataset_index": "n_unique_samples"}
    )
    pooled = pooled.merge(sample_counts, on=group_cols, how="left")
    seed = work.groupby(group_cols + ["seed"], as_index=False)[metric].median()
    center = seed.groupby(group_cols, as_index=False)[metric].agg(estimate="median", n_seeds="size")
    result = pooled.merge(center, on=group_cols, how="left")
    for seed_value, seed_frame in seed.groupby("seed"):
        column = _seed_name(seed_value)
        result = result.merge(
            seed_frame[group_cols + [metric]].rename(columns={metric: column}),
            on=group_cols, how="left",
        )
    result["metric"] = metric
    result["epsilon"] = EPSILON
    result["source_artifact"] = "; ".join(sorted(work["source_artifact"].dropna().unique()))
    result["coverage"] = float(work[metric].notna().mean())
    return result.sort_values(["model_key", "stage_order"]).reset_index(drop=True)


def _with_roles(frame: pd.DataFrame, mapping: Mapping[str, tuple[int, str]]) -> pd.DataFrame:
    work = frame[frame["layer"].isin(mapping)].copy()
    work["stage_order"] = work["layer"].map(lambda value: mapping[value][0])
    work["stage_label"] = work["layer"].map(lambda value: mapping[value][1])
    return work


def _plot_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "xtick.labelsize": 9.1, "ytick.labelsize": 9.2, "legend.fontsize": 8.8,
        "axes.linewidth": .8, "lines.linewidth": 2, "lines.markersize": 6,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def _clean_axis(axis, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D8DDE5", linewidth=.65, alpha=.7)
    axis.set_axisbelow(True)


def _save(figure, stem: Path) -> None:
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=360 if suffix == "png" else None,
                       bbox_inches="tight", pad_inches=.06)
    plt.close(figure)


def _plot_residual_routes(axis, work: pd.DataFrame, value: str, color: str,
                          linestyle: str, marker: str, label: str | None = None) -> None:
    by_layer = work.set_index("layer")
    common_layers = [
        "raw input", "conv 1", "pool 1", "conv 2", "encoder output",
        "interface dense", "pre-quantum latent",
    ]
    common = by_layer.loc[[layer for layer in common_layers if layer in by_layer.index]]
    axis.plot(common["stage_order"], common[value], color=color, linestyle=linestyle,
              marker=marker, markeredgecolor="#20242B", markeredgewidth=.4, label=label)
    split = by_layer.loc["pre-quantum latent"]
    merge = by_layer.loc["residual merge"]
    logits = by_layer.loc["final logits"]
    for branch, branch_marker in (("quantum branch", "o"), ("classical bypass", "s")):
        row = by_layer.loc[branch]
        axis.plot([split["stage_order"], row["stage_order"], merge["stage_order"], logits["stage_order"]],
                  [split[value], row[value], merge[value], logits[value]], color=color,
                  linestyle=linestyle, marker=branch_marker, markeredgecolor="#20242B",
                  markeredgewidth=.4, alpha=.95)


def plot_single(data: pd.DataFrame, stem: Path, metric: str, residual: bool = False) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    figure, axis = plt.subplots(figsize=(7.7, 4.55))
    work = data.sort_values("stage_order")
    key = str(work["model_key"].iloc[0])
    color = COLORS[key]
    if residual:
        _plot_residual_routes(axis, work, "estimate", color, "-", MARKERS[key])
        errors = np.vstack([work["estimate"] - work["q25"], work["q75"] - work["estimate"]])
        axis.errorbar(work["stage_order"], work["estimate"], yerr=errors, fmt="none",
                      ecolor=color, elinewidth=.75, capsize=2.2, alpha=.42)
    else:
        x = work["stage_order"].to_numpy(float)
        axis.fill_between(x, work["q25"], work["q75"], color=color, alpha=.15, linewidth=0)
        axis.plot(x, work["estimate"], color=color, marker=MARKERS[key],
                  markeredgecolor="#20242B", markeredgewidth=.4)
    axis.set_xticks(work["stage_order"].drop_duplicates(),
                    work.drop_duplicates("stage_order")["stage_label"], rotation=27, ha="right")
    axis.set_xlabel("Bloc architectural")
    if metric == "gain_rms":
        axis.set_ylabel(r"Gain RMS $G_\ell$")
        axis.set_yscale("log")
        axis.axhline(1, color="#343A40", linestyle=":", linewidth=.9)
    elif metric == "margin_normalized_harm":
        axis.set_ylabel(r"Dommage normalisé $\widetilde{H}_\ell$")
        axis.axhline(0, color="#343A40", linestyle="--", linewidth=.9)
        axis.axhline(1, color="#6B7280", linestyle=":", linewidth=.8)
    elif metric == "gradient_sensitivity":
        axis.set_ylabel(r"Sensibilité $S_\ell$")
        if (work["q25"] > 0).all(): axis.set_yscale("log")
    elif metric == "alignment_harmful":
        axis.set_ylabel(r"Alignement nuisible $C_\ell$")
        axis.axhline(0, color="#343A40", linestyle="--", linewidth=.9)
        axis.set_ylim(-1.03, 1.03)
    elif metric == "relative_displacement":
        axis.set_ylabel(r"Déplacement relatif $\|\Delta a_\ell\|_2/\|a_\ell\|_2$")
        if (work["q25"] > 0).all(): axis.set_yscale("log")
    _clean_axis(axis)
    figure.tight_layout()
    _save(figure, stem)


def plot_dual(data: pd.DataFrame, stem: Path, residual: bool = False) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    figure, left = plt.subplots(figsize=(7.9, 4.7))
    right = left.twinx()
    rms = data[data["metric"] == "gain_rms"].sort_values("stage_order")
    harm = data[data["metric"] == "margin_normalized_harm"].sort_values("stage_order")
    if residual:
        _plot_residual_routes(left, rms, "estimate", "#0072B2", "-", "o", "Gain RMS")
        _plot_residual_routes(right, harm, "estimate", "#D55E00", "--", "s", "Dommage normalisé")
        rms_errors = np.vstack([rms["estimate"] - rms["q25"], rms["q75"] - rms["estimate"]])
        harm_errors = np.vstack([harm["estimate"] - harm["q25"], harm["q75"] - harm["estimate"]])
        left.errorbar(rms["stage_order"], rms["estimate"], yerr=rms_errors, fmt="none",
                      ecolor="#0072B2", elinewidth=.75, capsize=2.2, alpha=.42)
        right.errorbar(harm["stage_order"], harm["estimate"], yerr=harm_errors, fmt="none",
                       ecolor="#D55E00", elinewidth=.75, capsize=2.2, alpha=.38)
    else:
        x = rms["stage_order"].to_numpy(float)
        left.fill_between(x, rms["q25"], rms["q75"], color="#0072B2", alpha=.15, linewidth=0)
        right.fill_between(x, harm["q25"], harm["q75"], color="#D55E00", alpha=.12, linewidth=0)
        left.plot(x, rms["estimate"], color="#0072B2", linestyle="-", marker="o",
                  markeredgecolor="#20242B", markeredgewidth=.4, label="Gain RMS")
        right.plot(x, harm["estimate"], color="#D55E00", linestyle="--", marker="s",
                   markeredgecolor="#20242B", markeredgewidth=.4, label="Dommage normalisé")
    left.set_yscale("log")
    left.axhline(1, color="#0072B2", linestyle=":", linewidth=.85, alpha=.75)
    right.axhline(0, color="#D55E00", linestyle="--", linewidth=.75, alpha=.72)
    right.axhline(1, color="#D55E00", linestyle=":", linewidth=.75, alpha=.65)
    labels = rms.drop_duplicates("stage_order")
    left.set_xticks(labels["stage_order"], labels["stage_label"], rotation=27, ha="right")
    left.set_xlabel("Bloc architectural")
    left.set_ylabel(r"Gain RMS $G_\ell$", color="#0072B2")
    right.set_ylabel(r"Dommage normalisé $\widetilde{H}_\ell$", color="#D55E00")
    left.tick_params(axis="y", colors="#0072B2")
    right.tick_params(axis="y", colors="#D55E00")
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    left.grid(axis="y", color="#D8DDE5", linewidth=.65, alpha=.65)
    lines = [
        plt.Line2D([], [], color="#0072B2", linestyle="-", marker="o", label="Gain RMS"),
        plt.Line2D([], [], color="#D55E00", linestyle="--", marker="s", label="Dommage normalisé"),
    ]
    left.legend(handles=lines, frameon=False, loc="best")
    figure.tight_layout()
    _save(figure, stem)


def plot_grouped(data: pd.DataFrame, stem: Path, metric: str, bands: bool = True,
                 direct_labels: bool = False, legend_outside: bool = False,
                 xticks: Sequence[float] | None = None,
                 xtick_labels: Sequence[str] | None = None,
                 xlabel: str = "Rôle architectural",
                 yaxis_side: str = "left") -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    figure, axis = plt.subplots(figsize=(8.25, 4.75))
    for key, work in data.groupby("model_key", sort=False):
        work = work.sort_values("stage_order")
        x = work["stage_order"].to_numpy(float)
        valid = np.isfinite(work["estimate"].to_numpy(float))
        if bands and valid.any():
            axis.fill_between(x, work["q25"], work["q75"], color=COLORS[key], alpha=.075, linewidth=0)
        axis.plot(x, work["estimate"], color=COLORS[key], linestyle=LINESTYLES[key],
                  marker=MARKERS[key], markeredgecolor="#20242B", markeredgewidth=.35,
                  label=str(work["model"].iloc[0]))
        if direct_labels and valid.any():
            last = work.loc[valid].iloc[-1]
            label_offsets = {
                "reshvql": 8, "mlp_large": -6, "mlpmixer": 8, "hpvql5": -7,
                "hvql": 8, "cnn": -8, "resnet18": 6, "mlp_small": -5,
                "hquanv": 5, "hvqm": 0, "hqe": 0,
            }
            axis.annotate(str(work["model"].iloc[0]), (last["stage_order"], last["estimate"]),
                          xytext=(5, label_offsets.get(str(key), 0)), textcoords="offset points", va="center",
                          fontsize=7.8, color=COLORS[key])
    if xticks is not None and xtick_labels is not None:
        axis.set_xticks(xticks, xtick_labels, rotation=22, ha="right")
    else:
        labels = data.sort_values("stage_order").drop_duplicates("stage_order")
        axis.set_xticks(labels["stage_order"], labels["stage_label"], rotation=22, ha="right")
    axis.set_xlabel(xlabel)
    if metric == "gain_rms":
        axis.set_ylabel(r"Gain RMS $G_\ell$")
        axis.set_yscale("log")
        axis.axhline(1, color="#343A40", linestyle=":", linewidth=.9)
    elif metric == "margin_normalized_harm":
        axis.set_ylabel(r"Dommage normalisé $\widetilde{H}_\ell$")
        axis.axhline(0, color="#343A40", linestyle="--", linewidth=.9)
        axis.axhline(1, color="#6B7280", linestyle=":", linewidth=.8)
    elif metric == "harm_efficiency":
        axis.set_ylabel(r"Efficacité $E_\ell=\widetilde{H}_\ell/(G_\ell+\tau_E)$")
        axis.axhline(0, color="#343A40", linestyle="--", linewidth=.9)
        axis.set_yscale("symlog", linthresh=.1, linscale=.8)
    elif metric == "relative_displacement":
        axis.set_ylabel("Déplacement relatif")
        axis.set_yscale("log")
    _clean_axis(axis)
    if yaxis_side == "left":
        axis.yaxis.set_label_position("left")
        axis.yaxis.tick_left()
        axis.spines["left"].set_visible(True)
        axis.spines["right"].set_visible(False)
    elif yaxis_side == "right":
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(True)
    if not direct_labels:
        if legend_outside:
            axis.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.01))
        else:
            axis.legend(frameon=False, ncol=2, loc="best")
    figure.tight_layout()
    _save(figure, stem)


def plot_interval(data: pd.DataFrame, stem: Path, metric: str) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    figure, axis = plt.subplots(figsize=(7.35, 4.35))
    work = data.sort_values("display_order").reset_index(drop=True)
    y = np.arange(len(work))
    for index, row in work.iterrows():
        color = COLORS[str(row["model_key"])]
        axis.plot([row["q25"], row["q75"]], [index, index], color=color, linewidth=5,
                  alpha=.27, solid_capstyle="round")
        axis.scatter(row["estimate"], index, color=color, marker=MARKERS[str(row["model_key"])],
                     edgecolor="#20242B", linewidth=.45, s=58, zorder=3)
    if metric != "delta_rms": axis.axvline(0, color="#343A40", linestyle="--", linewidth=.9)
    axis.set_yticks(y, work["comparison_label"])
    axis.invert_yaxis()
    labels = {
        "delta_rms": "Déplacement RMS au stade décisionnel",
        "gain_rms": r"Gain RMS $G$ au stade décisionnel",
        "margin_normalized_harm": r"Dommage normalisé $\widetilde{H}$",
        "harm_efficiency": r"Efficacité $\widetilde{H}/(G+\tau_E)$",
    }
    axis.set_xlabel(labels[metric])
    _clean_axis(axis, "x")
    figure.tight_layout()
    _save(figure, stem)


def plot_branch_bars(data: pd.DataFrame, stem: Path, metric: str) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    figure, axis = plt.subplots(figsize=(6.9, 4.25))
    work = data.sort_values("display_order")
    x = np.arange(len(work))
    colors = ["#0072B2", "#D55E00", "#6B7280"]
    errors = np.vstack([work["estimate"] - work["q25"], work["q75"] - work["estimate"]])
    axis.bar(x, work["estimate"], color=colors, edgecolor="#20242B", linewidth=.45)
    axis.errorbar(x, work["estimate"], yerr=errors, fmt="none", ecolor="#20242B", elinewidth=.8, capsize=3)
    if metric == "margin_normalized_harm": axis.axhline(0, color="#343A40", linestyle="--", linewidth=.9)
    axis.set_xticks(x, work["branch_fr"], rotation=16, ha="right")
    axis.set_ylabel("Déplacement RMS" if metric == "delta_rms" else r"Dommage normalisé $\widetilde{H}$")
    _clean_axis(axis)
    figure.tight_layout()
    _save(figure, stem)


def plot_retention(data: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    figure, axis = plt.subplots(figsize=(7.1, 4.3))
    work = data.sort_values("display_order")
    x = np.arange(len(work))
    errors = np.vstack([work["estimate"] - work["q25"], work["q75"] - work["estimate"]])
    colors = [COLORS[key] for key in work["model_key"]]
    axis.bar(x, work["estimate"], color=colors, edgecolor="#20242B", linewidth=.45)
    axis.errorbar(x, work["estimate"], yerr=errors, fmt="none", ecolor="#20242B", elinewidth=.8, capsize=3)
    axis.axhline(1, color="#343A40", linestyle=":", linewidth=.9)
    axis.set_xticks(x, work["model"], rotation=17, ha="right")
    axis.set_ylabel("Rétention RMS mesure / pré-interface")
    _clean_axis(axis)
    figure.tight_layout()
    _save(figure, stem)


def plot_paired_dual(data: pd.DataFrame, stem: Path) -> None:
    """Render HVQL and CNN side by side with only the two outer y axes."""
    import matplotlib.pyplot as plt
    _plot_style()
    figure, left_axes = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
    right_axes = []

    rms_all = data[data["metric"] == "gain_rms"]
    harm_all = data[data["metric"] == "margin_normalized_harm"]
    rms_low = max(float(rms_all["q25"].min()) * .72, 1e-3)
    rms_high = float(rms_all["q75"].max()) * 1.22
    harm_low = min(-.05, float(harm_all["q25"].min()) * 1.08)
    harm_high = max(1.05, float(harm_all["q75"].max()) * 1.08)

    for panel_index, (model_key, panel_label, left) in enumerate(
        zip(("hvql", "cnn"), ("HVQL", "CNN"), left_axes)
    ):
        panel = data[data["model_key"] == model_key]
        rms = panel[panel["metric"] == "gain_rms"].sort_values("stage_order")
        harm = panel[panel["metric"] == "margin_normalized_harm"].sort_values("stage_order")
        right = left.twinx()
        right_axes.append(right)
        x = rms["stage_order"].to_numpy(float)
        left.fill_between(x, rms["q25"], rms["q75"], color="#0072B2", alpha=.15, linewidth=0)
        right.fill_between(x, harm["q25"], harm["q75"], color="#D55E00", alpha=.12, linewidth=0)
        left.plot(x, rms["estimate"], color="#0072B2", linestyle="-", marker="o",
                  markeredgecolor="#20242B", markeredgewidth=.4)
        right.plot(x, harm["estimate"], color="#D55E00", linestyle="--", marker="s",
                   markeredgecolor="#20242B", markeredgewidth=.4)
        left.set_yscale("log")
        left.set_ylim(rms_low, rms_high)
        right.set_ylim(harm_low, harm_high)
        left.axhline(1, color="#0072B2", linestyle=":", linewidth=.85, alpha=.75)
        right.axhline(0, color="#D55E00", linestyle="--", linewidth=.75, alpha=.72)
        right.axhline(1, color="#D55E00", linestyle=":", linewidth=.75, alpha=.65)
        labels = rms.drop_duplicates("stage_order")
        left.set_xticks(labels["stage_order"], labels["stage_label"], rotation=27, ha="right")
        left.set_xlabel("")
        left.set_title(panel_label, fontsize=11.2, fontweight="semibold", pad=9)
        left.grid(axis="y", color="#D8DDE5", linewidth=.65, alpha=.65)
        left.spines["top"].set_visible(False)
        right.spines["top"].set_visible(False)

        if panel_index == 0:
            left.set_ylabel(r"Gain RMS $G_\ell$", color="#0072B2")
            left.tick_params(axis="y", colors="#0072B2")
            right.set_ylabel("")
            right.tick_params(axis="y", right=False, labelright=False)
            right.spines["right"].set_visible(False)
        else:
            left.set_ylabel("")
            left.tick_params(axis="y", left=False, labelleft=False)
            left.spines["left"].set_visible(False)
            right.set_ylabel(r"Dommage normalisé $\widetilde{H}_\ell$", color="#D55E00")
            right.tick_params(axis="y", colors="#D55E00")

    handles = [
        plt.Line2D([], [], color="#0072B2", linestyle="-", marker="o", label="Gain RMS"),
        plt.Line2D([], [], color="#D55E00", linestyle="--", marker="s", label="Dommage normalisé"),
    ]
    figure.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.01))
    figure.subplots_adjust(left=.075, right=.925, bottom=.25, top=.88, wspace=.08)
    _save(figure, stem)


def plot_by_contract(data: pd.DataFrame, stem: Path, contract: Mapping[str, object]) -> None:
    kind = str(contract["kind"])
    metric = str(contract.get("metric", ""))
    if kind == "dual": plot_dual(data, stem, bool(contract.get("residual", False)))
    elif kind == "single": plot_single(data, stem, metric, bool(contract.get("residual", False)))
    elif kind == "grouped": plot_grouped(
        data, stem, metric, bool(contract.get("bands", True)),
        bool(contract.get("direct_labels", False)), bool(contract.get("legend_outside", False)),
        contract.get("xticks"), contract.get("xtick_labels"),
        str(contract.get("xlabel", "Rôle architectural")),
        str(contract.get("yaxis_side", "left")),
    )
    elif kind == "interval": plot_interval(data, stem, metric)
    elif kind == "branch": plot_branch_bars(data, stem, metric)
    elif kind == "retention": plot_retention(data, stem)
    elif kind == "paired_dual": plot_paired_dual(data, stem)
    else: raise ValueError(f"Type de figure inconnu : {kind}")


def replot_figure(figure_id: str, data_csv: Path, output_stem: Path, contract_path: Path) -> None:
    contracts = json.loads(contract_path.read_text(encoding="utf-8"))
    plot_by_contract(pd.read_csv(data_csv), output_stem, contracts[figure_id])


def _figure_interpretation(figure_id: str) -> tuple[str, str, str]:
    if "dual_progression" in figure_id:
        return (
            "Le gain RMS et le dommage normalisé sont mesurés sur les mêmes exemples à chaque bloc.",
            "La figure met en évidence les transitions où la taille et la pertinence décisionnelle évoluent dans des directions différentes.",
            "Les deux axes indépendants permettent de comparer le sens des transitions, pas la pente visuelle ni l’échelle absolue des courbes.",
        )
    if "gradient_sensitivity" in figure_id:
        return (
            "La norme L2 du gradient de marge propre est suivie par bloc.",
            "Elle indique si une hausse du dommage est compatible avec une sensibilité aval accrue malgré un RMS stable ou décroissant.",
            "Une sensibilité élevée ne prouve ni causalité architecturale ni fragilité sans tenir compte de la direction du déplacement.",
        )
    if "alignment" in figure_id:
        return (
            "Le cosinus nuisible entre gradient de marge et déplacement est suivi avec son signe.",
            "Une hausse indique une meilleure compatibilité directionnelle avec une réduction locale de marge.",
            "L’alignement ne mesure pas la taille du déplacement et ne démontre pas une sélection proprement quantique.",
        )
    if "residual_hvql_branch" in figure_id:
        return (
            "Les contributions de la branche quantique, du contournement classique et de la fusion sont comparées en RMS ou en dommage normalisé.",
            "Le contournement transporte davantage de déplacement vers la fusion que la branche mesurée.",
            "Les gradients de branche et l’opération de fusion empêchent d’interpréter les barres comme des effets causaux indépendants.",
        )
    if "decision_routing" in figure_id:
        return (
            "Le bloc immédiatement antérieur à la décision est comparé sur une cohorte commun-correct diagnostique.",
            "La figure distingue traitement classique, quantum-assisted, quantum-dominant et fusion résiduelle.",
            "Les capacités, dimensions, marges et têtes diffèrent; il ne s’agit pas d’une ablation appariée.",
        )
    if "late_hybrid_interface" in figure_id or "hvql_hpvql" in figure_id:
        return (
            "Les valeurs sont mesurées avant l’interface, après mesure et aux logits, sans interpolation d’un stade absent.",
            "Le passage à la mesure peut réduire le gain RMS tout en concentrant le dommage normalisé.",
            "La figure est compatible avec une transmission directionnelle, mais n’isole pas causalement l’effet quantique du goulot et du readout.",
        )
    if "all_architectures" in figure_id:
        return (
            "Des couches réelles sont assignées à six rôles conceptuels explicitement documentés.",
            "La vue révèle des renversements et des trajectoires différentes sans forcer les étapes absentes.",
            "Cette comparaison exploratoire mélange familles, attaques propres et capacités; elle ne permet pas une conclusion causale globale.",
        )
    return (
        "La métrique est calculée au niveau échantillon puis résumée par médiane et intervalle interquartile.",
        "La progression localise les blocs où la taille ou le dommage se réorganise.",
        "La figure décrit une association interne et n’établit pas qu’un type de bloc cause la robustesse.",
    )


def _write_method(figure_id: str, record: Mapping[str, object], data: pd.DataFrame) -> Path:
    measured, visible, limitation = _figure_interpretation(figure_id)
    source = "; ".join(sorted(data.get("source_artifact", pd.Series(["trace CSV"])).dropna().astype(str).unique()))
    coverage = data.get("coverage", pd.Series([np.nan])).dropna()
    coverage_text = f"{100 * coverage.min():.2f} % à {100 * coverage.max():.2f} %" if not coverage.empty else "voir CSV"
    path = METHOD_DIR / f"{figure_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        f"# {figure_id}", "",
        f"- Modèles : {record['models']}",
        "- Checkpoints : correspondances prédictives validées dans `checkpoint_alignment.csv`; inversion historique HVQL/HVQM corrigée.",
        "- Attaque : FGSM blanche propre au modèle, ε = 0,10.",
        f"- Cohorte : {record['cohort']}",
        f"- Blocs : {record['stages']}",
        f"- Métrique : {record['metric']}",
        "- Normalisation : gain RMS par le RMS de la perturbation d’entrée; dommage par la marge propre du même échantillon avec τ = 10^-3.",
        "- Traitement des petites marges : exclusion si m_f(x) < 10^-3; aucune division d’une médiane agrégée.",
        "- Agrégation : calcul au niveau échantillon, médiane par graine, médiane des graines; IQR groupé des observations appariées.",
        f"- Couverture numérique : {coverage_text}.",
        f"- Source : {source}.",
        f"- Mesure directe : {measured}",
        f"- Lecture visible : {visible}",
        "- Mécanisme compatible : restructuration directionnelle du déplacement et concentration de sa pertinence aval.",
        f"- Ce que la figure ne prouve pas : {limitation}",
        "- Limites générales : attaques propres différentes, capacité, marge, largeur et optimisation non contrôlées.",
    ]) + "\n", encoding="utf-8")
    return path


def _caption(figure_id: str, record: Mapping[str, object]) -> str:
    return (
        f"\\texttt{{{figure_id}}} — {record['concept']}. Les points représentent les médianes et "
        "les intervalles les écarts interquartiles. FGSM blanche propre au modèle, "
        "$\\varepsilon=0{,}10$. Les valeurs sont calculées au niveau échantillon avant agrégation."
    )


def _write_replot_script(figure_id: str) -> Path:
    path = SCRIPT_DIR / f"{figure_id}_plot.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from pathlib import Path\nimport sys\n\n"
        "PACKAGE_ROOT = Path(__file__).resolve().parents[1]\n"
        "PROJECT_ROOT = Path(__file__).resolve().parents[3]\n"
        "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n"
        "from analytics.attacks.trace_figure_followup import replot_figure\n\n"
        f"FIGURE_ID = {figure_id!r}\n"
        "replot_figure(\n"
        "    FIGURE_ID,\n"
        "    PACKAGE_ROOT / 'followup_figure_data' / f'{FIGURE_ID}.csv',\n"
        "    PACKAGE_ROOT / 'followup_figures' / FIGURE_ID,\n"
        "    PACKAGE_ROOT / 'followup_plot_contracts.json',\n"
        ")\n",
        encoding="utf-8",
    )
    return path


def _record(records: list[dict[str, object]], contracts: dict[str, dict[str, object]],
            figure_id: str, data: pd.DataFrame, contract: dict[str, object], *,
            concept: str, models: str, metric: str, cohort: str, stages: str,
            recommended_use: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    data["plot_kind"] = contract["kind"]
    data.to_csv(DATA_DIR / f"{figure_id}.csv", index=False)
    plot_by_contract(data, FIGURE_DIR / figure_id, contract)
    contracts[figure_id] = contract
    script = _write_replot_script(figure_id)
    draft = {
        "concept": concept, "models": models, "metric": metric, "cohort": cohort,
        "stages": stages, "recommended_use": recommended_use,
    }
    method = _write_method(figure_id, draft, data)
    caption = _caption(figure_id, draft)
    caption_path = CAPTION_DIR / f"{figure_id}.tex"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    records.append({
        "figure_id": figure_id,
        "png_filename": f"followup_figures/{figure_id}.png",
        "pdf_filename": f"followup_figures/{figure_id}.pdf",
        "svg_filename": f"followup_figures/{figure_id}.svg",
        "source_csv": f"followup_figure_data/{figure_id}.csv",
        "plotting_script": script.relative_to(OUTPUT_ROOT).as_posix(),
        "method_file": method.relative_to(OUTPUT_ROOT).as_posix(),
        "caption_file": caption_path.relative_to(OUTPUT_ROOT).as_posix(),
        **draft,
        "status": "générée",
    })


def _prepare_hvql_cnn_pair(hvql: pd.DataFrame, cnn: pd.DataFrame) -> pd.DataFrame:
    """Map the saved hooks to the user-selected architectural progressions."""
    mappings = {
        "hvql": {
            "input": (0, "Entrée"),
            "conv1": (1, "Conv. 1"),
            "pool1": (2, "Pool 1"),
            "conv2": (3, "Conv. 2"),
            "encoder": (4, "Pool 2"),
            "interface_dense": (5, "Dense 1"),
            "prequantum": (6, "Interface Quantique"),
            "measurement": (7, "Mesures Quantiques"),
            "head": (8, "Dense 2"),
            "logits": (9, "Logits"),
        },
        "cnn": {
            "input": (0, "Entrée"),
            "conv1": (1, "Conv. 1"),
            "pool1": (2, "Pool 1"),
            "conv2": (3, "Conv. 2"),
            "encoder": (4, "Pool 2"),
            "predecision": (5, "Dense"),
            "logits": (6, "Logits"),
        },
    }
    chunks = []
    for key, source in (("hvql", hvql), ("cnn", cnn)):
        mapping = mappings[key]
        work = source[source["layer"].isin(mapping)].copy()
        work["stage_order"] = work["layer"].map(lambda layer: mapping[layer][0])
        work["stage_label"] = work["layer"].map(lambda layer: mapping[layer][1])
        work["panel_order"] = {"hvql": 0, "cnn": 1}[key]
        chunks.append(work)
    return pd.concat(chunks, ignore_index=True).sort_values(
        ["panel_order", "metric", "stage_order"]
    ).reset_index(drop=True)


def _per_model_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                       contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    outputs = {}
    slugs = {
        "hvql": "hvql_strong_d50", "cnn": "cnn_32_64_fc128",
        "mlp_large": "mlp_1024_512_64_64", "mlpmixer": "mlpmixer",
        "reshvql": "hvql_residual",
    }
    for key in ("hvql", "cnn", "mlp_large", "mlpmixer", "reshvql"):
        spec = MODEL_TRACES[key]
        frame = _model_frame(data, spec)
        summaries = {metric: summarize(frame, metric) for metric in ("gain_rms", "margin_normalized_harm")}
        dual = pd.concat(summaries.values(), ignore_index=True)
        outputs[f"{key}_dual"] = dual
        residual = key == "reshvql"
        common = dict(
            models=DISPLAY_LABELS[key], cohort=spec.evidence,
            stages="; ".join(frame.sort_values("stage_order")["stage_label"].drop_duplicates()),
        )
        _record(records, contracts, f"{slugs[key]}_dual_progression", dual,
                {"kind": "dual", "residual": residual},
                concept="Progression conjointe du gain RMS et du dommage normalisé",
                metric="gain RMS; dommage de marge normalisé par la marge propre",
                recommended_use="candidat principal" if key in {"hvql", "cnn"} else "annexe candidate", **common)
        for metric, suffix in (("gain_rms", "rms_progression"), ("margin_normalized_harm", "normalized_harm_progression")):
            _record(records, contracts, f"{slugs[key]}_{suffix}", summaries[metric],
                    {"kind": "single", "metric": metric, "residual": residual},
                    concept="Progression du gain RMS" if metric == "gain_rms" else "Progression du dommage normalisé",
                    metric=metric, recommended_use="contrôle de lisibilité", **common)

    pair_id = "hvql_cnn_dual_progression_side_by_side"
    pair = _prepare_hvql_cnn_pair(outputs["hvql_dual"], outputs["cnn_dual"])
    _record(
        records, contracts, pair_id, pair, {
            "kind": "paired_dual", "panel_order": ["hvql", "cnn"],
            "panel_labels": ["HVQL", "CNN"], "xlabel": "", "outer_y_axes_only": True,
        },
        concept="Progressions HVQL et CNN juxtaposées avec axes externes partagés",
        models="HVQL à gauche; CNN à droite",
        metric="gain RMS; dommage de marge normalisé par la marge propre",
        cohort="500 exemples par graine; graines 0 et 42",
        stages=(
            "HVQL : Entrée; Conv. 1; Pool 1; Conv. 2; Pool 2; Dense 1; Interface Quantique; "
            "Mesures Quantiques; Dense 2; Logits | CNN : Entrée; Conv. 1; Pool 1; Conv. 2; "
            "Pool 2; Dense; Logits"
        ),
        recommended_use="comparaison principale compacte",
    )
    outputs["hvql_cnn_dual_pair"] = pair
    return outputs


def _legacy_interface_frame(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    mappings = {
        "hvql": {"pre-quantum latent": (0, "Pré-interface"), "measured quantum vector": (1, "Mesures"), "final logits": (3, "Logits")},
        "hpvql": {"pre-quantum latent": (0, "Pré-interface"), "measured quantum vector": (1, "Mesures"), "final logits": (3, "Logits")},
        "hvqm": {"encoder output": (0, "Pré-interface"), "measured quantum logits": (1, "Mesures"), "final logits": (3, "Logits")},
        "hqe": {"pre-quantum latent": (0, "Pré-interface"), "measured quantum vector": (1, "Mesures"), "final logits": (3, "Logits")},
        "reshvql": {"pre-quantum latent": (0, "Pré-interface"), "quantum branch": (1, "Branche quantique"), "residual merge": (2, "Fusion résiduelle"), "final logits": (3, "Logits")},
    }
    key_map = {"hpvql": "hpvql5"}
    chunks = []
    legacy = data["legacy"]
    for raw_key, mapping in mappings.items():
        work = legacy[legacy["model_key"] == raw_key].copy()
        work["model_key"] = key_map.get(raw_key, raw_key)
        work["model"] = DISPLAY_LABELS[work["model_key"].iloc[0]]
        chunks.append(_with_roles(work, mapping))
    return pd.concat(chunks, ignore_index=True)


def _late_interface_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                            contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    frame = _legacy_interface_frame(data)
    outputs = {}
    for metric, figure_id in (
        ("gain_rms", "late_hybrid_interface_rms_progression"),
        ("margin_normalized_harm", "late_hybrid_interface_normalized_harm_progression"),
        ("harm_efficiency", "late_hybrid_interface_harm_efficiency_progression"),
    ):
        summary = summarize(frame, metric)
        contract: dict[str, object] = {"kind": "grouped", "metric": metric, "bands": True}
        if metric in {"gain_rms", "margin_normalized_harm"}:
            # Three shared ticks only.  The residual merge remains an observed
            # point halfway between the measurement and logits positions.
            summary["stage_order"] = summary["stage_order"].astype(float)
            summary.loc[summary["stage_order"] == 3, "stage_order"] = 2.0
            residual_merge = (summary["model_key"] == "reshvql") & (summary["layer"] == "residual merge")
            summary.loc[residual_merge, "stage_order"] = 1.5
            contract["xticks"] = [0.0, 1.0, 2.0]
            contract["xtick_labels"] = ["Pré-interface", "Mesures", "Logits"]
            contract["xlabel"] = ""
            summary.loc[summary["model_key"] == "hpvql5", "model"] = "HPVQL"
            summary.loc[summary["model_key"] == "reshvql", "model"] = "Résiduel"
        if metric == "margin_normalized_harm":
            contract["yaxis_side"] = "left"
        outputs[figure_id] = summary
        _record(records, contracts, figure_id, summary, contract,
                concept="Progression à l’interface des hybrides tardifs",
                models="HVQL; HPVQL; HVQM; HQE; HVQL résiduel",
                metric=metric, cohort="diagnostic commun-correct de 19 exemples; graine 0",
                stages="Pré-interface; mesures; fusion résiduelle représentée par un point intermédiaire sans graduation; logits",
                recommended_use="comparaison sélective" if metric != "harm_efficiency" else "diagnostic")
    return outputs


def _decision_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                      contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    legacy = data["legacy"]
    selection = {
        "cnn": ("dense ReLU", "CNN — représentation dense"),
        "hvql": ("measured quantum vector", "HVQL — mesures"),
        "hpvql": ("measured quantum vector", "HPVQL — mesures agrégées"),
        "hvqm": ("measured quantum logits", "HVQM — mesures décisionnelles"),
        "reshvql": ("residual merge", "HVQL résiduel — fusion"),
    }
    frames = []
    key_map = {"hpvql": "hpvql5"}
    for order, (raw_key, (layer, label)) in enumerate(selection.items()):
        work = legacy[(legacy["model_key"] == raw_key) & (legacy["layer"] == layer)].copy()
        work["model_key"] = key_map.get(raw_key, raw_key)
        work["model"] = DISPLAY_LABELS[work["model_key"].iloc[0]]
        work["comparison_label"] = label
        work["display_order"] = order
        work["stage_order"] = 0
        work["stage_label"] = label
        frames.append(work)
    frame = pd.concat(frames, ignore_index=True)
    outputs = {}
    for metric, figure_id in (
        ("gain_rms", "decision_routing_rms_comparison"),
        ("margin_normalized_harm", "decision_routing_normalized_harm_comparison"),
        ("harm_efficiency", "decision_routing_harm_efficiency_comparison"),
    ):
        summary = summarize(frame, metric)
        labels = frame[["model_key", "comparison_label", "display_order"]].drop_duplicates()
        summary = summary.merge(labels, on="model_key", how="left")
        if figure_id in {
            "decision_routing_harm_efficiency_comparison",
            "decision_routing_normalized_harm_comparison",
        }:
            summary["comparison_label"] = summary["model"]
        if figure_id == "decision_routing_normalized_harm_comparison":
            summary.loc[summary["model_key"] == "hpvql5", "comparison_label"] = "HPVQL"
            summary.loc[summary["model_key"] == "reshvql", "comparison_label"] = "Résiduel"
        outputs[figure_id] = summary
        _record(records, contracts, figure_id, summary, {"kind": "interval", "metric": metric},
                concept="Comparaison du routage juste avant la décision",
                models="CNN; HVQL; HPVQL; HVQM; HVQL résiduel", metric=metric,
                cohort="diagnostic apparié de 19 exemples commun-correct; graine 0",
                stages="représentation dense ou sortie mesurée/fusion immédiatement avant décision",
                recommended_use="comparaison de routage")

    residual = data["residual"].copy()
    residual["branch_fr"] = residual["branch_fr"].replace({
        "Branche quantique": "Branche quantique",
        "Contournement classique": "Contournement classique",
        "Fusion": "Fusion",
    })
    residual["display_order"] = residual["branch"].map({"quantum_branch": 0, "residual_branch": 1, "merge": 2})
    residual["stage_order"] = residual["display_order"]
    residual["stage_label"] = residual["branch_fr"]
    residual["layer"] = residual["branch"]
    residual["model"] = DISPLAY_LABELS["reshvql"]
    for metric, source_metric, figure_id in (
        ("delta_rms", "delta_rms_contribution", "residual_hvql_branch_rms_comparison"),
        ("margin_normalized_harm", "margin_normalized_harm", "residual_hvql_branch_harm_comparison"),
    ):
        summary = summarize(residual, source_metric)
        summary = summary.merge(residual[["model_key", "layer", "branch_fr", "display_order"]].drop_duplicates(), on=["model_key", "layer"], how="left")
        summary["metric"] = metric
        outputs[figure_id] = summary
        _record(records, contracts, figure_id, summary, {"kind": "branch", "metric": metric},
                concept="Décomposition des branches du HVQL résiduel", models="HVQL résiduel",
                metric=metric, cohort="500 exemples par graine; graines 0 et 42",
                stages="branche quantique; contournement classique mis à l’échelle; fusion",
                recommended_use="figure de soutien")
    return outputs


def _segmentation_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                          contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    mappings = {
        key: {"prequantum": (0, "Pré-interface"), "measurement": (1, "Mesures"), "logits": (2, "Logits")}
        for key in ("hvql", "hpvql5", "hpvql10")
    }
    chunks = []
    source = data["segmentation"]
    for key, mapping in mappings.items():
        work = source[source["model_key"] == key].copy()
        work["model"] = DISPLAY_LABELS[key]
        chunks.append(_with_roles(work, mapping))
    frame = pd.concat(chunks, ignore_index=True)
    outputs = {}
    for metric, figure_id in (
        ("gain_rms", "hvql_hpvql_rms_progression"),
        ("margin_normalized_harm", "hvql_hpvql_normalized_harm_progression"),
    ):
        summary = summarize(frame, metric)
        outputs[figure_id] = summary
        _record(records, contracts, figure_id, summary, {"kind": "grouped", "metric": metric, "bands": True},
                concept="Traitement quantique global ou partitionné", models="HVQL; HPVQL 5 blocs; HPVQL 10 blocs",
                metric=metric, cohort="500 exemples par graine; graines 0 et 42",
                stages="Pré-interface; mesures; logits", recommended_use="variation quantique contrôlée")
    rows = []
    for key, work in frame.groupby("model_key"):
        wide = work.pivot_table(index=["seed", "dataset_index"], columns="layer", values="delta_rms", aggfunc="first")
        ratio = wide["measurement"] / wide["prequantum"].replace(0, np.nan)
        seed_values = ratio.groupby(level="seed").median()
        rows.append({
            "model_key": key, "model": DISPLAY_LABELS[key], "display_order": {"hvql": 0, "hpvql5": 1, "hpvql10": 2}[key],
            "estimate": float(seed_values.median()), "q25": float(ratio.quantile(.25)), "q75": float(ratio.quantile(.75)),
            "n_unique_samples": int(ratio.notna().sum()), "n_seeds": int(len(seed_values)), "metric": "retention_rms",
            "source_artifact": "trace_segmentation_samples.csv", "coverage": float(ratio.notna().mean()),
        })
    retention = pd.DataFrame(rows)
    outputs["hvql_hpvql_interface_retention"] = retention
    _record(records, contracts, "hvql_hpvql_interface_retention", retention, {"kind": "retention"},
            concept="Rétention RMS de l’interface globale ou partitionnée",
            models="HVQL; HPVQL 5 blocs; HPVQL 10 blocs", metric="RMS mesure / RMS pré-interface",
            cohort="500 exemples par graine; graines 0 et 42", stages="pré-interface → mesures",
            recommended_use="variation quantique contrôlée principale")
    return outputs


def _routing_variants_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                              contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    mappings = {
        "hvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Conv. 1"), "conv 2": (2, "Conv. 2"), "encoder output": (3, "Encodeur"), "interface dense": (4, "Interface dense"), "pre-quantum latent": (5, "Latent"), "measured quantum vector": (6, "Sortie quantique"), "final logits": (8, "Logits")},
        "drophvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Conv. 1"), "conv 2": (2, "Conv. 2"), "encoder output": (3, "Encodeur"), "interface dense": (4, "Interface dense"), "pre-dropout latent": (5, "Latent"), "measured quantum vector": (6, "Sortie quantique"), "final logits": (8, "Logits")},
        "reshvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Conv. 1"), "conv 2": (2, "Conv. 2"), "encoder output": (3, "Encodeur"), "interface dense": (4, "Interface dense"), "pre-quantum latent": (5, "Latent"), "quantum branch": (6, "Branche quantique"), "residual merge": (7, "Fusion"), "final logits": (8, "Logits")},
    }
    chunks = []
    legacy = data["legacy"]
    for key, mapping in mappings.items():
        work = legacy[legacy["model_key"] == key].copy()
        work["model"] = DISPLAY_LABELS[key]
        chunks.append(_with_roles(work, mapping))
    frame = pd.concat(chunks, ignore_index=True)
    outputs = {}
    for metric, figure_id in (
        ("gain_rms", "hvql_routing_variants_rms_progression"),
        ("margin_normalized_harm", "hvql_routing_variants_normalized_harm_progression"),
    ):
        summary = summarize(frame, metric)
        outputs[figure_id] = summary
        _record(records, contracts, figure_id, summary, {"kind": "grouped", "metric": metric, "bands": True},
                concept="Routage standard, dropout et résiduel", models="HVQL standard; HVQL dropout; HVQL résiduel",
                metric=metric, cohort="diagnostic commun-correct de 19 exemples; graine 0",
                stages="étapes communes; branche quantique et fusion résiduelle explicitement séparées",
                recommended_use="annexe")
    return outputs


ALL_STAGE_MAPPING: Mapping[str, Mapping[str, tuple[int, str]]] = {
    "cnn": {"input": (0, "Entrée"), "conv1": (1, "Transformation initiale"), "encoder": (2, "Représentation intermédiaire"), "predecision": (3, "Représentation pré-décisionnelle"), "dense": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "resnet18": {"input": (0, "Entrée"), "stem": (1, "Transformation initiale"), "res2": (2, "Représentation intermédiaire"), "predecision": (3, "Représentation pré-décisionnelle"), "head": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "mlp_large": {"input": (0, "Entrée"), "dense1": (1, "Transformation initiale"), "dense2": (2, "Représentation intermédiaire"), "predecision": (3, "Représentation pré-décisionnelle"), "head": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "mlp_small": {"input": (0, "Entrée"), "dense1": (1, "Transformation initiale"), "predecision": (3, "Représentation pré-décisionnelle"), "head": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "mlpmixer": {"input": (0, "Entrée"), "patch_projection": (1, "Transformation initiale"), "mixer2": (2, "Représentation intermédiaire"), "predecision": (3, "Représentation pré-décisionnelle"), "head": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "hquanv": {"input": (0, "Entrée"), "quanv": (1, "Transformation initiale"), "encoder": (2, "Représentation intermédiaire"), "predecision": (3, "Représentation pré-décisionnelle"), "head": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "hvql": {"input": (0, "Entrée"), "conv1": (1, "Transformation initiale"), "encoder": (2, "Représentation intermédiaire"), "prequantum": (3, "Représentation pré-décisionnelle"), "measurement": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "hpvql5": {"input": (0, "Entrée"), "conv1": (1, "Transformation initiale"), "encoder": (2, "Représentation intermédiaire"), "prequantum": (3, "Représentation pré-décisionnelle"), "measurement": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "hvqm": {"input": (0, "Entrée"), "conv1": (1, "Transformation initiale"), "conv2": (2, "Représentation intermédiaire"), "encoder": (3, "Représentation pré-décisionnelle"), "measurement": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "hqe": {"input": (0, "Entrée"), "dense4096": (1, "Transformation initiale"), "encoder": (2, "Représentation intermédiaire"), "prequantum": (3, "Représentation pré-décisionnelle"), "measurement": (4, "Interface ou décision"), "logits": (5, "Logits")},
    "reshvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Transformation initiale"), "encoder output": (2, "Représentation intermédiaire"), "pre-quantum latent": (3, "Représentation pré-décisionnelle"), "residual merge": (4, "Interface ou décision"), "final logits": (5, "Logits")},
}


def _all_architecture_frame(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    primary_sources = {
        "cnn": "layerwise", "resnet18": "layerwise", "mlp_large": "layerwise", "mlp_small": "layerwise",
        "mlpmixer": "layerwise", "hquanv": "layerwise", "hvql": "interface", "hpvql5": "interface",
        "hvqm": "interface", "hqe": "interface", "reshvql": "legacy",
    }
    chunks = []
    for key, source in primary_sources.items():
        work = data[source]
        work = work[work["model_key"] == key].copy()
        work["model"] = DISPLAY_LABELS[key]
        chunks.append(_with_roles(work, ALL_STAGE_MAPPING[key]))
    return pd.concat(chunks, ignore_index=True)


def _all_stage_mapping_table() -> pd.DataFrame:
    rows = []
    for key, mapping in ALL_STAGE_MAPPING.items():
        assigned = {order: (layer, label) for layer, (order, label) in mapping.items()}
        for order, stage in enumerate(("Entrée", "Transformation initiale", "Représentation intermédiaire", "Représentation pré-décisionnelle", "Interface ou décision", "Logits")):
            layer, label = assigned.get(order, (np.nan, stage))
            rows.append({
                "model_key": key, "model": DISPLAY_LABELS[key], "conceptual_stage_order": order,
                "conceptual_stage": stage, "raw_traced_layer": layer,
                "status": "assigné" if isinstance(layer, str) else "manquant — aucune interpolation",
            })
    return pd.DataFrame(rows)


def _relative_all_frame(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    legacy = data["legacy"]
    legacy_maps = {
        "cnn": {"raw input": (0, "Entrée"), "conv 1": (1, "Transformation initiale"), "pool 2": (2, "Représentation intermédiaire"), "dense ReLU": (3, "Représentation pré-décisionnelle"), "linear head": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "resnet18": {"raw input": (0, "Entrée"), "stem convolution": (1, "Transformation initiale"), "residual stage 2": (2, "Représentation intermédiaire"), "global average pool": (3, "Représentation pré-décisionnelle"), "linear head": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "mlp_small": {"raw input": (0, "Entrée"), "dense 784->128": (1, "Transformation initiale"), "dense 128->64": (3, "Représentation pré-décisionnelle"), "linear head": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "mlpmixer": {"raw input": (0, "Entrée"), "patch projection": (1, "Transformation initiale"), "mixer block 2": (2, "Représentation intermédiaire"), "token normalization": (3, "Représentation pré-décisionnelle"), "linear head": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "hquanv": {"raw input": (0, "Entrée"), "quanvolution map": (1, "Transformation initiale"), "flattened quanvolution": (2, "Représentation intermédiaire"), "dense ReLU": (3, "Représentation pré-décisionnelle"), "linear head": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "hvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Transformation initiale"), "encoder output": (2, "Représentation intermédiaire"), "pre-quantum latent": (3, "Représentation pré-décisionnelle"), "measured quantum vector": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "hpvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Transformation initiale"), "encoder output": (2, "Représentation intermédiaire"), "pre-quantum latent": (3, "Représentation pré-décisionnelle"), "measured quantum vector": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "hvqm": {"raw input": (0, "Entrée"), "conv 1": (1, "Transformation initiale"), "conv 2": (2, "Représentation intermédiaire"), "encoder output": (3, "Représentation pré-décisionnelle"), "measured quantum logits": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "hqe": {"raw input": (0, "Entrée"), "dense 784->4096": (1, "Transformation initiale"), "encoder latent 1024": (2, "Représentation intermédiaire"), "pre-quantum latent": (3, "Représentation pré-décisionnelle"), "measured quantum vector": (4, "Interface ou décision"), "final logits": (5, "Logits")},
        "reshvql": {"raw input": (0, "Entrée"), "conv 1": (1, "Transformation initiale"), "encoder output": (2, "Représentation intermédiaire"), "pre-quantum latent": (3, "Représentation pré-décisionnelle"), "residual merge": (4, "Interface ou décision"), "final logits": (5, "Logits")},
    }
    chunks = []
    for raw_key, mapping in legacy_maps.items():
        work = legacy[legacy["model_key"] == raw_key].copy()
        key = "hpvql5" if raw_key == "hpvql" else raw_key
        work["model_key"] = key
        work["model"] = DISPLAY_LABELS[key]
        chunks.append(_with_roles(work, mapping))
    return pd.concat(chunks, ignore_index=True)


def _exploratory_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                         contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    frame = _all_architecture_frame(data)
    reduced = {"cnn", "mlp_large", "mlpmixer", "hvql", "hpvql5", "hvqm", "hqe", "reshvql"}
    outputs = {}
    configurations = [
        ("gain_rms", frame, "all_architectures_grouped_rms"),
        ("margin_normalized_harm", frame, "all_architectures_grouped_harm"),
        ("relative_displacement", _relative_all_frame(data), "all_architectures_grouped_relative_displacement"),
        ("harm_efficiency", frame, "all_architectures_grouped_harm_efficiency"),
    ]
    for metric, source, prefix in configurations:
        for group_name, subset in (("full", set(ALL_STAGE_MAPPING)), ("reduced", reduced)):
            selected = source[source["model_key"].isin(subset)].copy()
            summary = summarize(selected, metric)
            figure_id = f"{prefix}_{group_name}"
            outputs[figure_id] = summary
            _record(records, contracts, figure_id, summary,
                    {"kind": "grouped", "metric": metric, "bands": False, "direct_labels": False, "legend_outside": True},
                    concept="Vue exploratoire de toutes les architectures",
                    models="; ".join(DISPLAY_LABELS[key] for key in subset), metric=metric,
                    cohort="traces principales à 500 exemples/graine; HVQL résiduel et déplacement relatif sur diagnostic de 19 exemples",
                    stages="six rôles conceptuels; valeurs manquantes non interpolées", recommended_use="exploratoire")
            if metric in {"gain_rms", "margin_normalized_harm"}:
                iqr_id = f"{figure_id}_iqr"
                _record(records, contracts, iqr_id, summary,
                        {"kind": "grouped", "metric": metric, "bands": True, "direct_labels": False},
                        concept="Vue exploratoire avec incertitude",
                        models="; ".join(DISPLAY_LABELS[key] for key in subset), metric=metric,
                        cohort="mêmes traces; IQR groupé", stages="six rôles conceptuels; valeurs manquantes non interpolées",
                        recommended_use="exploratoire avec incertitude")
    return outputs


def _diagnostic_figures(data: Mapping[str, pd.DataFrame], records: list[dict[str, object]],
                        contracts: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    keys = ("hvql", "cnn", "mlp_large", "mlpmixer", "reshvql")
    slugs = {"hvql": "hvql", "cnn": "cnn", "mlp_large": "mlp_1024_512_64_64", "mlpmixer": "mlpmixer", "reshvql": "hvql_residual"}
    outputs = {}
    for key in keys:
        frame = _model_frame(data, MODEL_TRACES[key])
        residual = key == "reshvql"
        for metric, suffix in (("gradient_sensitivity", "gradient_sensitivity_progression"), ("alignment_harmful", "harmful_alignment_progression")):
            summary = summarize(frame, metric)
            figure_id = f"{slugs[key]}_{suffix}"
            outputs[figure_id] = summary
            recommendation = "alignement" if key in {"hvql", "mlpmixer", "reshvql"} else "sensibilité et alignement conjointement"
            _record(records, contracts, figure_id, summary,
                    {"kind": "single", "metric": metric, "residual": residual},
                    concept="Diagnostic de sensibilité" if metric == "gradient_sensitivity" else "Diagnostic d’alignement nuisible",
                    models=DISPLAY_LABELS[key], metric=metric, cohort=MODEL_TRACES[key].evidence,
                    stages="blocs réels du modèle", recommended_use=f"annexe; diagnostic recommandé : {recommendation}")
    return outputs


def _normalization_coverage(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for source in ("layerwise", "interface", "segmentation", "residual", "legacy"):
        frame = data[source]
        for (model_key, seed), work in frame.groupby(["model_key", "seed"], sort=False):
            unique = work.drop_duplicates("dataset_index")
            rows.append({
                "source": source, "model_key": model_key,
                "model": DISPLAY_LABELS.get(str(model_key), str(model_key)), "seed": int(seed),
                "n_samples": int(len(unique)),
                "n_covered": int(unique["margin_normalization_covered"].sum()),
                "coverage": float(unique["margin_normalization_covered"].mean()),
                "min_clean_margin": float(unique["clean_margin"].min()),
                "median_clean_margin": float(unique["clean_margin"].median()),
                "tau": MARGIN_TAU, "threshold": MARGIN_THRESHOLD,
            })
    return pd.DataFrame(rows)


def _unavailable() -> pd.DataFrame:
    return pd.DataFrame([
        {"requested_figures": "hvql_ansatz_*", "status": "non générées",
         "reason": "Aucune trace couche par couche n’existe pour HVQL basic et random; les logits d’attaque ne permettent pas de reconstruire les activations internes.",
         "needed": "Traces appariées strong/basic/random sur les mêmes blocs, exemples et graines."},
        {"requested_figures": "hvql_depth_*", "status": "non générées",
         "reason": "Les traces qdepth=100/200 sauvegardées concernent VQM, pas HVQL; les logits propres HVQL ne suffisent pas à reconstruire la progression.",
         "needed": "Traces HVQL strong aux profondeurs 50, 100 et 200."},
        {"requested_figures": "all_architectures_grouped_relative_displacement — grand MLP", "status": "valeur manquante",
         "reason": "La norme de l’activation propre n’a pas été sauvegardée pour le grand MLP; aucune interpolation n’est appliquée.",
         "needed": "Norme des activations propres par couche pour le grand MLP."},
    ])


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame([
        {"rank": 1, "role": "Figure principale de progression", "figure": "hvql_strong_d50_dual_progression.pdf",
         "demonstrates": "La divergence bloc par bloc entre gain RMS et dommage normalisé dans HVQL.",
         "placement": "Corps principal, ouverture du récit mécanistique.",
         "required_text": "Préciser que les axes indépendants autorisent la comparaison du sens des transitions, pas des pentes.",
         "too_strong": "La couche quantique débruite ou supprime la perturbation."},
        {"rank": 2, "role": "Comparaison classique", "figure": "cnn_32_64_fc128_dual_progression.pdf",
         "demonstrates": "La concentration du dommage dans une chaîne convolutionnelle sans interface quantique.",
         "placement": "Corps principal, immédiatement après HVQL.",
         "required_text": "Présenter CNN comme comparaison structurale, non comme ablation de capacité appariée.",
         "too_strong": "La différence avec HVQL est causée par la couche quantique."},
        {"rank": 3, "role": "Interface sélective", "figure": "late_hybrid_interface_normalized_harm_progression.pdf",
         "demonstrates": "Le dommage évolue différemment avant mesure, après mesure et aux logits selon le routage hybride.",
         "placement": "Corps principal, section sur l’interface.",
         "required_text": "Indiquer le diagnostic commun-correct de 19 exemples et la fusion résiduelle explicite.",
         "too_strong": "La mesure quantique est causalement protectrice."},
        {"rank": 4, "role": "Routage décisionnel", "figure": "decision_routing_harm_efficiency_comparison.pdf",
         "demonstrates": "La pertinence décisionnelle par unité de gain diffère entre dense classique, mesures et fusion résiduelle.",
         "placement": "Corps principal ou transition vers l’annexe mécanistique.",
         "required_text": "Rappeler la petite cohorte et les différences de capacité et de marge.",
         "too_strong": "L’efficacité mesure la responsabilité quantique pure."},
        {"rank": 5, "role": "Variation quantique contrôlée", "figure": "hvql_hpvql_interface_retention.pdf",
         "demonstrates": "Le couplage global complet n’est pas nécessaire pour obtenir une rétention RMS comparable à l’interface.",
         "placement": "Corps principal comme figure de soutien ou annexe proche.",
         "required_text": "Préciser que paramètres et dimensions mesurées ne sont pas strictement appariés.",
         "too_strong": "La segmentation est protectrice."},
        {"rank": 6, "role": "Vue toutes architectures", "figure": "all_architectures_grouped_rms_reduced.pdf",
         "demonstrates": "Exploration des renversements de gain sur six rôles grossiers.",
         "placement": "Annexe seulement; utilisable comme carte exploratoire.",
         "required_text": "Joindre le fichier de mapping et signaler les cohortes et capacités hétérogènes.",
         "too_strong": "Le graphique établit un classement causal des architectures."},
    ])


def _inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file():
            rows.append({"path": path.relative_to(OUTPUT_ROOT).as_posix(), "extension": path.suffix.lower(), "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def refresh_requested_layout_corrections(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    """Refresh only the figures affected by the final label/tick corrections."""
    global OUTPUT_ROOT, FIGURE_DIR, DATA_DIR, METHOD_DIR, CAPTION_DIR, SCRIPT_DIR
    OUTPUT_ROOT = output_root
    FIGURE_DIR = output_root / "followup_figures"
    DATA_DIR = output_root / "followup_figure_data"
    METHOD_DIR = output_root / "followup_methods"
    CAPTION_DIR = output_root / "followup_captions"
    SCRIPT_DIR = output_root / "followup_scripts"

    contract_path = output_root / "followup_plot_contracts.json"
    contracts = json.loads(contract_path.read_text(encoding="utf-8"))
    updated: list[str] = []

    decision_id = "decision_routing_harm_efficiency_comparison"
    decision_path = DATA_DIR / f"{decision_id}.csv"
    decision = pd.read_csv(decision_path)
    decision["comparison_label"] = decision["model"]
    decision.to_csv(decision_path, index=False)
    plot_by_contract(decision, FIGURE_DIR / decision_id, contracts[decision_id])
    updated.append(decision_id)

    harm_id = "decision_routing_normalized_harm_comparison"
    harm_path = DATA_DIR / f"{harm_id}.csv"
    harm = pd.read_csv(harm_path)
    harm["comparison_label"] = harm["model"]
    harm.loc[harm["model_key"] == "hpvql5", "comparison_label"] = "HPVQL"
    harm.loc[harm["model_key"] == "reshvql", "comparison_label"] = "Résiduel"
    harm.to_csv(harm_path, index=False)
    plot_by_contract(harm, FIGURE_DIR / harm_id, contracts[harm_id])
    updated.append(harm_id)

    late_ids = (
        "late_hybrid_interface_rms_progression",
        "late_hybrid_interface_normalized_harm_progression",
    )
    stage_text = (
        "Pré-interface; mesures; fusion résiduelle représentée par un point "
        "intermédiaire sans graduation; logits"
    )
    for figure_id in late_ids:
        data_path = DATA_DIR / f"{figure_id}.csv"
        figure_data = pd.read_csv(data_path)
        figure_data["stage_order"] = figure_data["stage_order"].astype(float)
        figure_data.loc[figure_data["layer"] == "final logits", "stage_order"] = 2.0
        residual_merge = (figure_data["model_key"] == "reshvql") & (figure_data["layer"] == "residual merge")
        figure_data.loc[residual_merge, "stage_order"] = 1.5
        figure_data.loc[figure_data["model_key"] == "hpvql5", "model"] = "HPVQL"
        figure_data.loc[figure_data["model_key"] == "reshvql", "model"] = "Résiduel"
        figure_data.to_csv(data_path, index=False)
        contracts[figure_id]["xticks"] = [0.0, 1.0, 2.0]
        contracts[figure_id]["xtick_labels"] = ["Pré-interface", "Mesures", "Logits"]
        contracts[figure_id]["xlabel"] = ""
        if figure_id == "late_hybrid_interface_normalized_harm_progression":
            contracts[figure_id]["yaxis_side"] = "left"
        plot_by_contract(figure_data, FIGURE_DIR / figure_id, contracts[figure_id])

        method_path = METHOD_DIR / f"{figure_id}.md"
        method_text = method_path.read_text(encoding="utf-8")
        method_text = re.sub(
            r"^- Blocs :.*$",
            f"- Blocs : {stage_text}",
            method_text,
            count=1,
            flags=re.MULTILINE,
        )
        method_path.write_text(method_text, encoding="utf-8")
        updated.append(figure_id)

    pair_id = "hvql_cnn_dual_progression_side_by_side"
    hvql = pd.read_csv(DATA_DIR / "hvql_strong_d50_dual_progression.csv")
    cnn = pd.read_csv(DATA_DIR / "cnn_32_64_fc128_dual_progression.csv")
    pair = _prepare_hvql_cnn_pair(hvql, cnn)
    pair_records: list[dict[str, object]] = []
    _record(
        pair_records, contracts, pair_id, pair, {
            "kind": "paired_dual", "panel_order": ["hvql", "cnn"],
            "panel_labels": ["HVQL", "CNN"], "xlabel": "", "outer_y_axes_only": True,
        },
        concept="Progressions HVQL et CNN juxtaposées avec axes externes partagés",
        models="HVQL à gauche; CNN à droite",
        metric="gain RMS; dommage de marge normalisé par la marge propre",
        cohort="500 exemples par graine; graines 0 et 42",
        stages=(
            "HVQL : Entrée; Conv. 1; Pool 1; Conv. 2; Pool 2; Dense 1; Interface Quantique; "
            "Mesures Quantiques; Dense 2; Logits | CNN : Entrée; Conv. 1; Pool 1; Conv. 2; "
            "Pool 2; Dense; Logits"
        ),
        recommended_use="comparaison principale compacte",
    )
    caption_path = CAPTION_DIR / f"{pair_id}.tex"
    caption_path.write_text(
        "\\texttt{hvql_cnn_dual_progression_side_by_side} — Progression conjointe de HVQL "
        "(à gauche) et du CNN (à droite). Le gain RMS utilise l’axe externe gauche et le dommage "
        "normalisé l’axe externe droit; les deux panneaux partagent les mêmes échelles. Les points "
        "représentent les médianes et les bandes les intervalles interquartiles, pour une FGSM blanche "
        "propre au modèle à $\\varepsilon=0{,}10$.\n",
        encoding="utf-8",
    )
    updated.append(pair_id)

    contract_path.write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output_root / "followup_figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest["figure_id"].isin(late_ids), "stages"] = stage_text
    manifest = manifest[manifest["figure_id"] != pair_id]
    manifest = pd.concat([manifest, pd.DataFrame(pair_records)], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    captions = []
    for path in sorted(CAPTION_DIR.glob("*.tex")):
        captions.extend([f"% {path.stem}", path.read_text(encoding="utf-8").strip(), ""])
    (output_root / "followup_captions_fr.tex").write_text("\n".join(captions), encoding="utf-8")
    summary_path = output_root / "followup_generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "figure_count": int(len(manifest)),
        "png_count": len(list(FIGURE_DIR.glob("*.png"))),
        "pdf_count": len(list(FIGURE_DIR.glob("*.pdf"))),
        "svg_count": len(list(FIGURE_DIR.glob("*.svg"))),
        "source_csv_count": len(list(DATA_DIR.glob("*.csv"))),
        "method_count": len(list(METHOD_DIR.glob("*.md"))),
        "plot_script_count": len(list(SCRIPT_DIR.glob("*_plot.py"))),
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copy2(__file__, SCRIPT_DIR / "trace_figure_followup.py")
    refresh_script = PROJECT_ROOT / "scripts" / "refresh_trace_figure_followup_layout.py"
    if refresh_script.exists():
        shutil.copy2(refresh_script, SCRIPT_DIR / refresh_script.name)
    _inventory().to_csv(output_root / "followup_output_inventory.csv", index=False)
    return {"updated_figure_ids": updated, "updated_count": len(updated)}


def build_followup_package(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    global OUTPUT_ROOT, FIGURE_DIR, DATA_DIR, METHOD_DIR, CAPTION_DIR, SCRIPT_DIR, NOTEBOOK_DIR
    OUTPUT_ROOT = output_root
    FIGURE_DIR = output_root / "followup_figures"
    DATA_DIR = output_root / "followup_figure_data"
    METHOD_DIR = output_root / "followup_methods"
    CAPTION_DIR = output_root / "followup_captions"
    SCRIPT_DIR = output_root / "followup_scripts"
    NOTEBOOK_DIR = output_root / "followup_notebooks"
    for directory in (FIGURE_DIR, DATA_DIR, METHOD_DIR, CAPTION_DIR, SCRIPT_DIR, NOTEBOOK_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts()
    data = prepared_data(artifacts)
    records: list[dict[str, object]] = []
    contracts: dict[str, dict[str, object]] = {}

    _per_model_figures(data, records, contracts)
    _late_interface_figures(data, records, contracts)
    _decision_figures(data, records, contracts)
    _segmentation_figures(data, records, contracts)
    _routing_variants_figures(data, records, contracts)
    _exploratory_figures(data, records, contracts)
    _diagnostic_figures(data, records, contracts)

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_root / "followup_figure_manifest.csv", index=False)
    metric_definition_audit().to_csv(output_root / "metric_definition_audit.csv", index=False)
    _normalization_coverage(data).to_csv(output_root / "margin_normalization_coverage.csv", index=False)
    _all_stage_mapping_table().to_csv(output_root / "all_architectures_stage_mapping.csv", index=False)
    _unavailable().to_csv(output_root / "followup_unavailable_figures.csv", index=False)
    _recommendations().to_csv(output_root / "followup_figure_recommendations.csv", index=False)
    discover_sources().to_csv(output_root / "followup_source_inventory.csv", index=False)
    (output_root / "followup_plot_contracts.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")
    captions = []
    for path in sorted(CAPTION_DIR.glob("*.tex")):
        captions.append(f"% {path.stem}")
        captions.append(path.read_text(encoding="utf-8").strip())
        captions.append("")
    (output_root / "followup_captions_fr.tex").write_text("\n".join(captions), encoding="utf-8")

    shutil.copy2(__file__, SCRIPT_DIR / "trace_figure_followup.py")
    driver = PROJECT_ROOT / "scripts" / "generate_trace_figure_followup.py"
    if driver.exists(): shutil.copy2(driver, SCRIPT_DIR / driver.name)
    notebook = PROJECT_ROOT / "src" / "analytics" / "attacks" / "trace_figure_followup.ipynb"
    if notebook.exists(): shutil.copy2(notebook, NOTEBOOK_DIR / notebook.name)
    _inventory().to_csv(output_root / "followup_output_inventory.csv", index=False)

    result = {
        "figure_count": int(len(manifest)),
        "png_count": len(list(FIGURE_DIR.glob("*.png"))),
        "pdf_count": len(list(FIGURE_DIR.glob("*.pdf"))),
        "svg_count": len(list(FIGURE_DIR.glob("*.svg"))),
        "source_csv_count": len(list(DATA_DIR.glob("*.csv"))),
        "method_count": len(list(METHOD_DIR.glob("*.md"))),
        "plot_script_count": len([path for path in SCRIPT_DIR.glob("*_plot.py")]),
        "unavailable_requested_groups": int(len(_unavailable())),
        "normalization_tau": MARGIN_TAU,
        "normalization_threshold": MARGIN_THRESHOLD,
    }
    (output_root / "followup_generation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
