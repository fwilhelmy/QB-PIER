"""Trace-backed redesign of mechanistic adversarial-propagation figures.

The module is intentionally library-only.  It never loads a checkpoint and it
never generates a new attack: every exported value comes from the CSV artifacts
created by the earlier tracing runs.  The companion notebook and the driver in
``scripts/`` call :func:`build_redesigned_figure_package`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import shutil
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure_redesign_from_trace"
NEW_TRACE_DIR = PROJECT_ROOT / "outputs" / "mechanistic_visualizations"
OLD_TRACE_DIR = PROJECT_ROOT / "outputs" / "mechanistic_propagation"
PRIMARY_EPSILON = 0.10
NORMALIZATION_TAU = 1e-3
DETERMINISTIC_SEED = 20260727


@dataclass(frozen=True)
class ModelView:
    key: str
    label: str
    slug: str
    source: str
    layers: tuple[str, ...]
    evidence: str


MODEL_VIEWS: tuple[ModelView, ...] = (
    ModelView("hvql", "HVQL | strong | d=50", "hvql_d50", "layerwise", (
        "input", "conv1", "pool1", "conv2", "encoder", "interface_dense",
        "prequantum", "measurement", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("hpvql5", "HPVQL | strong | 5 blocs | d=50", "hpvql_5blocks_d50", "interface", (
        "input", "conv1", "pool1", "conv2", "encoder", "interface_dense",
        "prequantum", "measurement", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("hvqm", "HVQM | strong | d=50", "hvqm_d50", "interface", (
        "input", "conv1", "pool1", "conv2", "encoder", "measurement", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("hqe", "HQE | baseline", "hqe_baseline", "interface", (
        "input", "dense4096", "relu4096", "dense2048", "relu2048", "dense1024",
        "encoder", "interface_dense", "prequantum", "measurement", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("reshvql", "HVQL | résiduel", "hvql_residual", "legacy", (
        "raw input", "conv 1", "pool 1", "conv 2", "encoder output", "interface dense",
        "pre-quantum latent", "quantum branch", "classical bypass", "residual merge", "final logits",
    ), "diagnostic de 19 exemples; graine 0"),
    ModelView("drophvql", "HVQL | dropout", "hvql_dropout", "legacy", (
        "raw input", "conv 1", "pool 1", "conv 2", "encoder output", "interface dense",
        "pre-dropout latent", "inference dropout", "measured quantum vector", "final logits",
    ), "diagnostic de 19 exemples; graine 0"),
    ModelView("cnn", "CNN | conv=[32,64], fc=[128]", "cnn_strong", "layerwise", (
        "input", "conv1", "pool1", "conv2", "encoder", "dense", "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("mlp_large", "MLP | [1024,512,64,64]", "mlp_large", "layerwise", (
        "input", "dense1", "dense2", "dense3", "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42; modèle non attaqué dans le balayage historique"),
    ModelView("mlp_small", "MLP | [128,64]", "mlp_small", "layerwise", (
        "input", "dense1", "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("resnet18", "ResNet18", "resnet18", "layerwise", (
        "input", "stem", "stem_relu", "res1", "res2", "res3", "encoder", "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("mlpmixer", "MLPMixer", "mlpmixer", "layerwise", (
        "input", "patch_projection", "patch_tokens", "mixer1", "mixer2", "mixer3",
        "encoder", "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
    ModelView("hquanv", "HQuanv", "hquanv", "layerwise", (
        "input", "quanv", "encoder", "dense", "predecision", "logits",
    ), "500 exemples par graine; graines 0 et 42"),
)


OLD_LAYER_LABELS: Mapping[str, str] = {
    "raw input": "Entrée",
    "conv 1": "Conv. 1",
    "pool 1": "Pool 1",
    "conv 2": "Conv. 2",
    "pool 2": "Pool 2",
    "dense hidden": "Projection dense",
    "dense ReLU": "Activation dense",
    "final logits": "Logits",
    "stem convolution": "Tige conv.",
    "stem activation": "Activation tige",
    "residual stage 1": "Bloc résiduel 1",
    "residual stage 2": "Bloc résiduel 2",
    "residual stage 3": "Bloc résiduel 3",
    "residual stage 4": "Bloc résiduel 4",
    "global average pool": "Moyenne globale",
    "linear head": "Lecture linéaire",
    "dense 784->128": "Dense 784→128",
    "dense 128->64": "Dense 128→64",
    "patch projection": "Projection patchs",
    "patch tokens": "Jetons patchs",
    "mixer block 1": "Mixer 1",
    "mixer block 2": "Mixer 2",
    "mixer block 3": "Mixer 3",
    "mixer block 4": "Mixer 4",
    "token normalization": "Normalisation",
    "encoder output": "Encodeur",
    "interface dense": "Interface dense",
    "pre-quantum latent": "Latent préquantique",
    "measured quantum vector": "Mesures quantiques",
    "classical head": "Tête classique",
    "measured quantum logits": "Logits quantiques",
    "dense 784->4096": "Dense 784→4096",
    "ReLU 4096": "ReLU 4096",
    "dense 4096->2048": "Dense 4096→2048",
    "ReLU 2048": "ReLU 2048",
    "dense 2048->1024": "Dense 2048→1024",
    "encoder latent 1024": "Latent 1024",
    "quanvolution map": "Quanvolution",
    "flattened quanvolution": "Carte quanvolution",
    "pre-dropout latent": "Latent pré-dropout",
    "inference dropout": "Dropout (inférence)",
    "quantum branch": "Branche quantique",
    "classical bypass": "Contournement classique",
    "residual merge": "Fusion résiduelle",
}


COLORS: Mapping[str, str] = {
    "hvql": "#0072B2",
    "hpvql5": "#009E73",
    "hpvql10": "#56B4E9",
    "hvqm": "#AA4499",
    "hqe": "#999933",
    "reshvql": "#D55E00",
    "drophvql": "#CC79A7",
    "cnn": "#0072B2",
    "mlp_small": "#E69F00",
    "mlp_large": "#D55E00",
    "resnet18": "#332288",
    "mlpmixer": "#CC79A7",
    "hquanv": "#117733",
}

MARKERS: Mapping[str, str] = {
    "hvql": "o", "hpvql5": "s", "hpvql10": "^", "hvqm": "D", "hqe": "P",
    "reshvql": "X", "drophvql": "v", "cnn": "o", "mlp_small": "^",
    "mlp_large": "v", "resnet18": "s", "mlpmixer": "D", "hquanv": "P",
}

LINESTYLES: Mapping[str, str] = {
    "hvql": "-", "hpvql5": "--", "hpvql10": ":", "hvqm": "-.", "hqe": ":",
    "reshvql": "--", "drophvql": "-.", "cnn": "-", "mlp_small": "-.",
    "mlp_large": ":", "resnet18": "--", "mlpmixer": "-", "hquanv": "--",
}


def source_paths() -> dict[str, Path]:
    """Return the immutable tracing artifacts used by the redesign."""

    return {
        "layerwise": NEW_TRACE_DIR / "trace_layerwise_samples.csv",
        "interface": NEW_TRACE_DIR / "trace_interface_samples.csv",
        "segmentation": NEW_TRACE_DIR / "trace_segmentation_samples.csv",
        "interface_transitions": NEW_TRACE_DIR / "interface_transitions.csv",
        "residual": NEW_TRACE_DIR / "trace_residual_samples.csv",
        "legacy": OLD_TRACE_DIR / "layerwise_trace_samples.csv",
        "variant_metrics": OLD_TRACE_DIR / "variant_seed_metrics.csv",
        "checkpoint_alignment": NEW_TRACE_DIR / "checkpoint_alignment.csv",
        "architecture_audit": NEW_TRACE_DIR / "architecture_label_audit.csv",
    }


def discover_trace_artifacts() -> pd.DataFrame:
    """Inventory required artifacts without loading model checkpoints."""

    rows = []
    for key, path in source_paths().items():
        rows.append({
            "source_key": key,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def load_trace_artifacts() -> dict[str, pd.DataFrame]:
    """Load only the CSV evidence needed for the requested redesign."""

    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trace artifacts: " + ", ".join(missing))
    return {key: pd.read_csv(path) for key, path in paths.items()}


def _prepare_new_trace(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[np.isclose(frame["epsilon"], PRIMARY_EPSILON)].copy()
    work = work[work["condition"] == "auto-FGSM"].copy()
    work["relative_displacement"] = np.nan
    work["evidence_source"] = "trace_layerwise_samples.csv"
    return work


def _prepare_old_trace(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[
        np.isclose(frame["epsilon"], PRIMARY_EPSILON) & (frame["condition"] == "self")
    ].copy()
    work["seed"] = 0
    work["dataset_index"] = work["sample"].astype(int)
    work["layer_fr"] = work["layer"].map(OLD_LAYER_LABELS).fillna(work["layer"])
    work["projected_margin_harm"] = work["predicted_margin_drop"]
    work["relative_displacement"] = work["relative_l2"]
    work["alignment_harmful"] = -work["margin_gradient_alignment"]
    input_harm = (
        work[work["layer"] == "raw input"]
        .set_index(["model_key", "dataset_index"])["projected_margin_harm"]
    )
    index = pd.MultiIndex.from_frame(work[["model_key", "dataset_index"]])
    denominator = input_harm.reindex(index).to_numpy(float)
    work["normalized_margin_harm"] = (
        work["projected_margin_harm"].to_numpy(float)
        / (np.abs(denominator) + NORMALIZATION_TAU)
    )
    work["evidence_source"] = "layerwise_trace_samples.csv"
    return work


def prepared_traces(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize the column contract across the two saved tracing runs."""

    layerwise = _prepare_new_trace(artifacts["layerwise"])
    interface = _prepare_new_trace(artifacts["interface"])
    interface["evidence_source"] = "trace_interface_samples.csv"
    segmentation = _prepare_new_trace(artifacts["segmentation"])
    segmentation["evidence_source"] = "trace_segmentation_samples.csv"
    legacy = _prepare_old_trace(artifacts["legacy"])
    return {
        "layerwise": layerwise,
        "interface": interface,
        "segmentation": segmentation,
        "legacy": legacy,
    }


def _model_view(key: str) -> ModelView:
    return next(view for view in MODEL_VIEWS if view.key == key)


def model_trace(traces: Mapping[str, pd.DataFrame], view: ModelView) -> pd.DataFrame:
    """Select the primary saved trace and meaningful stages for one model."""

    frame = traces[view.source]
    work = frame[frame["model_key"] == view.key].copy()
    order = {layer: index for index, layer in enumerate(view.layers)}
    work = work[work["layer"].isin(order)].copy()
    work["stage_order"] = work["layer"].map(order).astype(int)
    work["stage_label"] = work["layer_fr"]
    work["model"] = view.label
    return work.sort_values(["stage_order", "seed", "dataset_index"])


def _seed_column(seed: object) -> str:
    try:
        return f"seed_{int(seed)}_median"
    except (TypeError, ValueError):
        return f"seed_{seed}_median"


def summarize_progression(
    frame: pd.DataFrame,
    metric: str,
    *,
    model_key: str | None = None,
    model_label: str | None = None,
    stage_order: str = "stage_order",
    stage_label: str = "stage_label",
) -> pd.DataFrame:
    """Aggregate paired rows into pooled IQRs and median seed trajectories."""

    work = frame.dropna(subset=[metric]).copy()
    if work.empty:
        return pd.DataFrame()
    stage_cols = [stage_order, stage_label, "layer"]
    pooled = (
        work.groupby(stage_cols, as_index=False)[metric]
        .agg(q25=lambda x: x.quantile(0.25), pooled_median="median", q75=lambda x: x.quantile(0.75), n_samples="size")
    )
    seed_values = (
        work.groupby(stage_cols + ["seed"], as_index=False)[metric]
        .median()
    )
    estimates = (
        seed_values.groupby(stage_cols, as_index=False)[metric]
        .agg(estimate="median", n_seeds="size")
    )
    result = pooled.merge(estimates, on=stage_cols, how="left")
    for seed, seed_frame in seed_values.groupby("seed"):
        seed_frame = seed_frame[stage_cols + [metric]].rename(columns={metric: _seed_column(seed)})
        result = result.merge(seed_frame, on=stage_cols, how="left")
    result["model_key"] = model_key if model_key is not None else str(work["model_key"].iloc[0])
    result["model"] = model_label if model_label is not None else str(work["model"].iloc[0])
    result["metric"] = metric
    result["epsilon"] = PRIMARY_EPSILON
    result["attack"] = "FGSM blanche propre au modèle"
    result["source_artifact"] = str(work["evidence_source"].iloc[0]) if "evidence_source" in work else "trace CSV"
    return result.sort_values(stage_order).reset_index(drop=True)


def per_model_progression_data(traces: Mapping[str, pd.DataFrame]) -> dict[tuple[str, str], pd.DataFrame]:
    """Create the three requested per-model progression datasets when supported."""

    outputs: dict[tuple[str, str], pd.DataFrame] = {}
    for view in MODEL_VIEWS:
        work = model_trace(traces, view)
        for metric in ("delta_rms", "projected_margin_harm"):
            outputs[(view.key, metric)] = summarize_progression(
                work, metric, model_key=view.key, model_label=view.label
            )
        legacy = traces["legacy"]
        relative = legacy[legacy["model_key"] == ("mlp" if view.key == "mlp_small" else view.key)].copy()
        if view.key == "hpvql5":
            relative = legacy[legacy["model_key"] == "hpvql"].copy()
        if view.key == "mlp_large":
            relative = pd.DataFrame()
        if not relative.empty:
            layer_order = {layer: index for index, layer in enumerate(view.layers)}
            if view.source != "legacy":
                old_equivalents = _old_layer_selection(view.key)
                layer_order = {layer: index for index, layer in enumerate(old_equivalents)}
            relative = relative[relative["layer"].isin(layer_order)].copy()
            relative["stage_order"] = relative["layer"].map(layer_order).astype(int)
            relative["stage_label"] = relative["layer_fr"]
            relative["model_key"] = view.key
            relative["model"] = view.label
            outputs[(view.key, "relative_displacement")] = summarize_progression(
                relative, "relative_displacement", model_key=view.key, model_label=view.label
            )
    return outputs


def _old_layer_selection(model_key: str) -> tuple[str, ...]:
    selections: Mapping[str, tuple[str, ...]] = {
        "cnn": ("raw input", "conv 1", "pool 1", "conv 2", "pool 2", "dense hidden", "dense ReLU", "final logits"),
        "resnet18": ("raw input", "stem convolution", "stem activation", "residual stage 1", "residual stage 2", "residual stage 3", "residual stage 4", "global average pool", "final logits"),
        "mlp_small": ("raw input", "dense 784->128", "dense 128->64", "final logits"),
        "mlpmixer": ("raw input", "patch projection", "patch tokens", "mixer block 1", "mixer block 2", "mixer block 3", "mixer block 4", "token normalization", "final logits"),
        "hquanv": ("raw input", "quanvolution map", "flattened quanvolution", "dense hidden", "dense ReLU", "final logits"),
        "hvql": ("raw input", "conv 1", "pool 1", "conv 2", "encoder output", "interface dense", "pre-quantum latent", "measured quantum vector", "final logits"),
        "hpvql5": ("raw input", "conv 1", "pool 1", "conv 2", "encoder output", "interface dense", "pre-quantum latent", "measured quantum vector", "final logits"),
        "hvqm": ("raw input", "conv 1", "pool 1", "conv 2", "encoder output", "measured quantum logits", "final logits"),
        "hqe": ("raw input", "dense 784->4096", "ReLU 4096", "dense 4096->2048", "ReLU 2048", "dense 2048->1024", "encoder latent 1024", "interface dense", "pre-quantum latent", "measured quantum vector", "final logits"),
        "drophvql": _model_view("drophvql").layers,
        "reshvql": _model_view("reshvql").layers,
    }
    return selections.get(model_key, tuple())


def grouped_role_data(
    frame: pd.DataFrame,
    models: Sequence[str],
    mapping: Mapping[str, Mapping[str, tuple[int, str]]],
    metric: str,
) -> pd.DataFrame:
    """Aggregate model-specific raw layers on a declared architectural-role axis."""

    chunks = []
    for key in models:
        model_frame = frame[frame["model_key"] == key].copy()
        role_map = mapping[key]
        model_frame = model_frame[model_frame["layer"].isin(role_map)].copy()
        model_frame["stage_order"] = model_frame["layer"].map(lambda x: role_map[x][0])
        model_frame["stage_label"] = model_frame["layer"].map(lambda x: role_map[x][1])
        label = _display_label(key)
        chunks.append(summarize_progression(model_frame, metric, model_key=key, model_label=label))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def late_hybrid_grouped_data(traces: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    mapping = {
        "hvql": {
            "input": (0, "Entrée"), "conv1": (1, "Encodeur initial"),
            "encoder": (2, "Encodeur profond"), "prequantum": (3, "Pré-interface quantique"),
            "measurement": (4, "Mesures quantiques"), "logits": (5, "Logits"),
        },
        "hpvql5": {
            "input": (0, "Entrée"), "conv1": (1, "Encodeur initial"),
            "encoder": (2, "Encodeur profond"), "prequantum": (3, "Pré-interface quantique"),
            "measurement": (4, "Mesures quantiques"), "logits": (5, "Logits"),
        },
        "hvqm": {
            "input": (0, "Entrée"), "conv1": (1, "Encodeur initial"),
            "encoder": (3, "Pré-interface quantique"),
            "measurement": (4, "Mesures quantiques"), "logits": (5, "Logits"),
        },
    }
    result = grouped_role_data(traces["interface"], ("hvql", "hpvql5", "hvqm"), mapping, metric)
    # HVQM has no distinct deep-encoder stage.  Insert an explicit gap so the
    # plotted path does not imply that such a block was traced.
    hvqm_gap = {
        "stage_order": 2,
        "stage_label": "Encodeur profond",
        "layer": "non applicable",
        "q25": np.nan,
        "pooled_median": np.nan,
        "q75": np.nan,
        "n_samples": 0,
        "estimate": np.nan,
        "n_seeds": 0,
        "model_key": "hvqm",
        "model": "HVQM",
        "metric": metric,
        "epsilon": PRIMARY_EPSILON,
        "attack": "FGSM blanche propre au modèle",
        "source_artifact": "trace_interface_samples.csv",
    }
    return pd.concat([result, pd.DataFrame([hvqm_gap])], ignore_index=True).sort_values(
        ["model_key", "stage_order"]
    )


def segmentation_progression_data(traces: Mapping[str, pd.DataFrame], models: Sequence[str], metric: str) -> pd.DataFrame:
    frame = traces["segmentation"]
    chunks = []
    layer_order = {
        layer: index for index, layer in enumerate((
            "input", "conv1", "pool1", "conv2", "encoder", "interface_dense",
            "prequantum", "measurement", "logits",
        ))
    }
    for key in models:
        work = frame[(frame["model_key"] == key) & frame["layer"].isin(layer_order)].copy()
        work["stage_order"] = work["layer"].map(layer_order).astype(int)
        work["stage_label"] = work["layer_fr"]
        chunks.append(summarize_progression(work, metric, model_key=key, model_label=_display_label(key)))
    return pd.concat(chunks, ignore_index=True)


def legacy_variant_progression_data(traces: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    frame = traces["legacy"]
    mapping = {
        "hvql": {
            "raw input": (0, "Entrée"), "conv 1": (1, "Conv. 1"),
            "encoder output": (2, "Encodeur"), "interface dense": (3, "Interface dense"),
            "pre-quantum latent": (4, "Pré-interface"), "measured quantum vector": (5, "Sortie quantique"),
            "final logits": (7, "Logits"),
        },
        "drophvql": {
            "raw input": (0, "Entrée"), "conv 1": (1, "Conv. 1"),
            "encoder output": (2, "Encodeur"), "interface dense": (3, "Interface dense"),
            "pre-dropout latent": (4, "Pré-interface"),
            "measured quantum vector": (5, "Sortie quantique"), "final logits": (7, "Logits"),
        },
        "reshvql": {
            "raw input": (0, "Entrée"), "conv 1": (1, "Conv. 1"),
            "encoder output": (2, "Encodeur"), "interface dense": (3, "Interface dense"),
            "pre-quantum latent": (4, "Pré-interface"), "quantum branch": (5, "Sortie quantique"),
            "residual merge": (6, "Fusion résiduelle"), "final logits": (7, "Logits"),
        },
    }
    return grouped_role_data(frame, ("hvql", "drophvql", "reshvql"), mapping, metric)


def decision_stage_data(traces: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Compare matched decision-stage representations on the legacy 19-image cohort."""

    frame = traces["legacy"]
    selection = {
        "cnn": ("dense ReLU", "CNN — activation dense"),
        "hvqm": ("measured quantum logits", "HVQM — sortie quantique"),
        "hvql": ("measured quantum vector", "HVQL — mesures quantiques"),
        "reshvql": ("residual merge", "HVQL résiduel — fusion"),
    }
    rows = []
    for key, (layer, label) in selection.items():
        work = frame[(frame["model_key"] == key) & (frame["layer"] == layer)].dropna(subset=[metric])
        values = work[metric]
        rows.append({
            "model_key": key,
            "model": label,
            "layer": layer,
            "estimate": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "n_samples": int(len(values)),
            "n_seeds": 1,
            "metric": metric,
            "epsilon": PRIMARY_EPSILON,
            "source_artifact": "layerwise_trace_samples.csv",
        })
    return pd.DataFrame(rows)


def logits_margin_data(traces: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = traces["legacy"]
    labels = {
        "cnn": "CNN — activation dense",
        "hvqm": "HVQM — sortie quantique",
        "hvql": "HVQL — mesures quantiques",
        "reshvql": "HVQL résiduel — fusion",
    }
    rows = []
    for key, label in labels.items():
        work = frame[(frame["model_key"] == key) & (frame["layer"] == "final logits")]
        for column, state in (("clean_margin", "Propre"), ("adv_margin", "Adversaire")):
            values = work[column].dropna()
            rows.append({
                "model_key": key, "model": label, "state": state,
                "estimate": float(values.median()), "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)), "n_samples": int(len(values)),
                "metric": "decision_margin", "epsilon": PRIMARY_EPSILON,
                "source_artifact": "layerwise_trace_samples.csv",
            })
    return pd.DataFrame(rows)


def interface_summary_data(artifacts: Mapping[str, pd.DataFrame], traces: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    transitions = artifacts["interface_transitions"]
    transitions = transitions[np.isclose(transitions["epsilon"], PRIMARY_EPSILON)].copy()
    logits = traces["interface"]
    logits = logits[logits["layer"] == "logits"][[
        "model_key", "seed", "dataset_index", "normalized_margin_harm",
    ]].rename(columns={"normalized_margin_harm": "logits_harm_concentration"})
    merged = transitions.merge(logits, on=["model_key", "seed", "dataset_index"], how="left")
    rows = []
    for key, model_frame in merged.groupby("model_key", sort=False):
        pre_rms_column = "delta_rms__encoder" if key == "hvqm" else "delta_rms__prequantum"
        seed_rows = []
        for seed, seed_frame in model_frame.groupby("seed"):
            seed_rows.append({
                "seed": int(seed),
                "pre_rms": float(seed_frame[pre_rms_column].median()),
                "measured_rms": float(seed_frame["delta_rms__measurement"].median()),
                "retention_rms": float(seed_frame["retention_rms"].median()),
                "interface_harm_change": float(seed_frame["harm_change_normalized"].median()),
                "logits_harm_concentration": float(seed_frame["logits_harm_concentration"].median()),
                "alignment_change": float(seed_frame["alignment_change"].median()),
            })
        seed_df = pd.DataFrame(seed_rows)
        row = {
            "model_key": key,
            "model": _display_label(key),
            "n_samples": int(len(model_frame)),
            "n_seeds": int(model_frame["seed"].nunique()),
            "epsilon": PRIMARY_EPSILON,
            "source_artifact": "interface_transitions.csv + trace_interface_samples.csv",
        }
        for column in ("pre_rms", "measured_rms", "retention_rms", "interface_harm_change", "logits_harm_concentration", "alignment_change"):
            row[column] = float(seed_df[column].median())
            for seed_row in seed_rows:
                row[f"seed_{seed_row['seed']}_{column}"] = seed_row[column]
        pooled_columns = {
            "pre_rms": pre_rms_column,
            "measured_rms": "delta_rms__measurement",
            "retention_rms": "retention_rms",
            "interface_harm_change": "harm_change_normalized",
            "logits_harm_concentration": "logits_harm_concentration",
            "alignment_change": "alignment_change",
        }
        for output_column, source_column in pooled_columns.items():
            values = model_frame[source_column].dropna()
            row[f"{output_column}_q25"] = float(values.quantile(0.25))
            row[f"{output_column}_q75"] = float(values.quantile(0.75))
        rows.append(row)
    order = {key: index for index, key in enumerate(("hvql", "hpvql5", "hvqm", "hqe"))}
    result = pd.DataFrame(rows)
    result["model_order"] = result["model_key"].map(order)
    return result.sort_values("model_order").reset_index(drop=True)


def residual_branch_data(artifacts: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    frame = artifacts["residual"]
    frame = frame[np.isclose(frame["epsilon"], PRIMARY_EPSILON)].copy()
    rows = []
    for branch, work in frame.groupby("branch", sort=False):
        seed_values = work.groupby("seed")[metric].median()
        values = work[metric].dropna()
        row = {
            "branch": branch,
            "branch_fr": str(work["branch_fr"].iloc[0]),
            "estimate": float(seed_values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "n_samples": int(len(values)),
            "n_seeds": int(work["seed"].nunique()),
            "metric": metric,
            "epsilon": PRIMARY_EPSILON,
            "source_artifact": "trace_residual_samples.csv",
        }
        for seed, value in seed_values.items():
            row[_seed_column(seed)] = float(value)
        rows.append(row)
    order = {"quantum_branch": 0, "classical_bypass_scaled": 1, "merged": 2}
    result = pd.DataFrame(rows)
    result["order"] = result["branch"].map(order)
    return result.sort_values("order").reset_index(drop=True)


def _display_label(key: str) -> str:
    labels = {
        "hvql": "HVQL",
        "hpvql5": "HPVQL (5 blocs)",
        "hpvql10": "HPVQL (10 blocs)",
        "hvqm": "HVQM",
        "hqe": "HQE",
        "reshvql": "HVQL résiduel",
        "drophvql": "HVQL dropout",
        "cnn": "CNN",
        "mlp_small": "MLP [128,64]",
        "mlp_large": "MLP [1024,512,64,64]",
        "resnet18": "ResNet18",
        "mlpmixer": "MLPMixer",
        "hquanv": "HQuanv",
    }
    return labels.get(key, key)


def _plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.4,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _clean_axes(axis, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D8DDE5", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)


def _save_triplet(figure, output_stem: Path) -> dict[str, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for suffix in ("pdf", "svg", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=360 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.06)
        paths[suffix] = path
    __import__("matplotlib.pyplot").pyplot.close(figure)
    return paths


def plot_progression(data: pd.DataFrame, output_stem: Path, metric: str, *, residual_routing: bool = False) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(7.5, 4.45))
    key = str(data["model_key"].iloc[0])
    color = COLORS[key]
    ordered = data.sort_values("stage_order")
    x = ordered["stage_order"].to_numpy(float)
    if residual_routing and key == "reshvql":
        _plot_residual_progression(axis, ordered, color)
    else:
        axis.fill_between(x, ordered["q25"], ordered["q75"], color=color, alpha=0.16, linewidth=0)
        for column in [item for item in ordered.columns if item.startswith("seed_") and item.endswith("_median")]:
            axis.plot(x, ordered[column], color=color, alpha=0.32, linewidth=0.9, linestyle="--")
        axis.plot(x, ordered["estimate"], color=color, marker=MARKERS[key], markeredgecolor="#20242B", markeredgewidth=0.45)
    axis.set_xticks(x, ordered["stage_label"], rotation=28, ha="right")
    axis.set_xlabel("Bloc architectural")
    if metric == "delta_rms":
        axis.set_ylabel(r"Déplacement RMS $\mathrm{RMS}(\Delta a_\ell)$")
        if (ordered["q25"] > 0).all():
            axis.set_yscale("log")
    elif metric == "projected_margin_harm":
        axis.set_ylabel(r"Dommage de marge projeté $H_\ell$")
        axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
    else:
        axis.set_ylabel(r"Déplacement relatif $\|\Delta a_\ell\|_2 / \|a_\ell\|_2$")
        if (ordered["q25"] > 0).all():
            axis.set_yscale("log")
    _clean_axes(axis)
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def _plot_residual_progression(axis, ordered: pd.DataFrame, color: str) -> None:
    """Draw a branch-aware HVQL-residual path without implying branch sequence."""

    by_layer = ordered.set_index("layer")
    common = [layer for layer in _model_view("reshvql").layers if layer not in {"quantum branch", "classical bypass", "residual merge", "final logits"}]
    common = [layer for layer in common if layer in by_layer.index]
    common_frame = by_layer.loc[common]
    axis.fill_between(common_frame["stage_order"], common_frame["q25"], common_frame["q75"], color=color, alpha=0.14, linewidth=0)
    axis.plot(common_frame["stage_order"], common_frame["estimate"], color=color, marker="o", markeredgecolor="#20242B", markeredgewidth=0.4)
    split_x = float(common_frame["stage_order"].iloc[-1])
    split_y = float(common_frame["estimate"].iloc[-1])
    branch_x = float(by_layer.loc["quantum branch", "stage_order"])
    merge_x = float(by_layer.loc["residual merge", "stage_order"])
    merge_y = float(by_layer.loc["residual merge", "estimate"])
    branch_styles = {
        "quantum branch": ("#0072B2", "o", -0.08),
        "classical bypass": ("#D55E00", "s", 0.08),
    }
    for layer, (branch_color, marker, offset) in branch_styles.items():
        row = by_layer.loc[layer]
        bx = branch_x + offset
        by = float(row["estimate"])
        axis.plot([split_x, bx, merge_x], [split_y, by, merge_y], color=branch_color, linewidth=1.8, linestyle="--" if layer == "classical bypass" else "-")
        axis.scatter([bx], [by], color=branch_color, marker=marker, edgecolor="#20242B", linewidth=0.45, zorder=4, label=row["stage_label"])
    tail = by_layer.loc[[layer for layer in ("residual merge", "final logits") if layer in by_layer.index]]
    axis.plot(tail["stage_order"], tail["estimate"], color=color, marker="X", markeredgecolor="#20242B", markeredgewidth=0.45)
    axis.legend(frameon=False, loc="best")


def plot_grouped_progression(data: pd.DataFrame, output_stem: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(7.8, 4.55))
    for key, work in data.groupby("model_key", sort=False):
        work = work.sort_values("stage_order")
        x = work["stage_order"].to_numpy(float)
        color = COLORS[key]
        axis.fill_between(x, work["q25"], work["q75"], color=color, alpha=0.10, linewidth=0)
        axis.plot(
            x, work["estimate"], color=color, marker=MARKERS[key], linestyle=LINESTYLES[key],
            markeredgecolor="#20242B", markeredgewidth=0.4, label=str(work["model"].iloc[0]),
        )
    labels = data.sort_values("stage_order").drop_duplicates("stage_order")
    axis.set_xticks(labels["stage_order"], labels["stage_label"], rotation=24, ha="right")
    axis.set_xlabel("Rôle architectural")
    if metric == "delta_rms":
        axis.set_ylabel(r"Déplacement RMS $\mathrm{RMS}(\Delta a_\ell)$")
        if (data["q25"] > 0).all():
            axis.set_yscale("log")
    elif metric == "normalized_margin_harm":
        axis.set_ylabel(r"Dommage de marge normalisé $\widetilde{H}_\ell$")
        axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
    elif metric == "projected_margin_harm":
        axis.set_ylabel(r"Dommage de marge projeté $H_\ell$")
        axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
    elif metric == "alignment_harmful":
        axis.set_ylabel(r"Alignement nuisible $C_\ell$")
        axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
        axis.set_ylim(-1.03, 1.03)
    _clean_axes(axis)
    axis.legend(frameon=False, ncol=2, loc="best")
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def plot_horizontal_interval(data: pd.DataFrame, output_stem: Path, ylabel: str, *, zero_line: bool = False) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(7.1, 4.15))
    ordered = data.reset_index(drop=True)
    y = np.arange(len(ordered))
    for index, row in ordered.iterrows():
        color = COLORS[str(row["model_key"])]
        axis.plot([row["q25"], row["q75"]], [index, index], color=color, linewidth=5.0, alpha=0.30, solid_capstyle="round")
        axis.scatter([row["estimate"]], [index], color=color, marker=MARKERS[str(row["model_key"])], edgecolor="#20242B", linewidth=0.5, s=58, zorder=3)
    if zero_line:
        axis.axvline(0, color="#343A40", linewidth=0.9, linestyle="--")
    axis.set_yticks(y, ordered["model"])
    axis.invert_yaxis()
    axis.set_xlabel(ylabel)
    _clean_axes(axis, grid_axis="x")
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def plot_margin_pairs(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(7.5, 4.35))
    models = data["model"].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.34
    styles = {"Propre": ("#4D5968", -width / 2), "Adversaire": ("#D55E00", width / 2)}
    for state, (color, offset) in styles.items():
        work = data[data["state"] == state].set_index("model").loc[models]
        errors = np.vstack([work["estimate"] - work["q25"], work["q75"] - work["estimate"]])
        axis.bar(x + offset, work["estimate"], width, color=color, edgecolor="#20242B", linewidth=0.45, label=state)
        axis.errorbar(x + offset, work["estimate"], yerr=errors, fmt="none", ecolor="#20242B", elinewidth=0.8, capsize=2.5)
    axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
    axis.set_xticks(x, models, rotation=20, ha="right")
    axis.set_ylabel("Marge de décision aux logits")
    axis.legend(frameon=False, ncol=2)
    _clean_axes(axis)
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def plot_interface_transition(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(7.4, 4.45))
    x = np.array([0.0, 1.0])
    for _, row in data.iterrows():
        key = str(row["model_key"])
        y = np.array([row["pre_rms"], row["measured_rms"]], dtype=float)
        axis.plot(x, y, color=COLORS[key], marker=MARKERS[key], linestyle=LINESTYLES[key], label=row["model"], markeredgecolor="#20242B", markeredgewidth=0.4)
        lower = np.array([row["pre_rms_q25"], row["measured_rms_q25"]], dtype=float)
        upper = np.array([row["pre_rms_q75"], row["measured_rms_q75"]], dtype=float)
        axis.errorbar(
            x, y, yerr=np.vstack([y - lower, upper - y]), fmt="none",
            ecolor=COLORS[key], elinewidth=0.8, capsize=2.5, alpha=0.55,
        )
        axis.annotate(
            rf"$\widetilde{{H}}_{{\mathrm{{logits}}}}={row['logits_harm_concentration']:.2f}$",
            (1.0, y[1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.3, color=COLORS[key],
        )
    axis.set_xticks(x, ["Pré-interface", "Mesures quantiques"])
    axis.set_ylabel(r"Déplacement RMS $\mathrm{RMS}(\Delta a)$")
    axis.set_yscale("log")
    axis.legend(frameon=False, ncol=2, loc="best")
    _clean_axes(axis)
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def plot_retention_harm(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(6.8, 4.55))
    axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
    axis.axvline(1, color="#343A40", linewidth=0.9, linestyle=":")
    offsets = {"hvql": (7, 8), "hpvql5": (7, -14), "hvqm": (7, 7), "hqe": (7, 7)}
    for _, row in data.iterrows():
        key = str(row["model_key"])
        x_error = np.array([[row["retention_rms"] - row["retention_rms_q25"]], [row["retention_rms_q75"] - row["retention_rms"]]])
        y_error = np.array([[row["interface_harm_change"] - row["interface_harm_change_q25"]], [row["interface_harm_change_q75"] - row["interface_harm_change"]]])
        axis.errorbar(
            row["retention_rms"], row["interface_harm_change"], xerr=x_error, yerr=y_error,
            fmt="none", ecolor=COLORS[key], elinewidth=0.75, capsize=2.2, alpha=0.45,
        )
        axis.scatter(row["retention_rms"], row["interface_harm_change"], color=COLORS[key], marker=MARKERS[key], edgecolor="#20242B", linewidth=0.5, s=68)
        axis.annotate(row["model"], (row["retention_rms"], row["interface_harm_change"]), xytext=offsets[key], textcoords="offset points", fontsize=8.8)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=0.25, linscale=0.7)
    axis.set_xlabel("Rétention RMS mesure / pré-interface")
    axis.set_ylabel(r"Variation du dommage normalisé $\Delta\widetilde{H}$")
    _clean_axes(axis)
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def plot_residual_interface_comparison(data: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(7.5, 4.35))
    categories = ["Pré-interface", "Mesure / branche q", "Bypass classique", "Fusion", "Logits"]
    x = np.arange(len(categories))
    for key, work in data.groupby("model_key", sort=False):
        work = work.sort_values("stage_order")
        axis.plot(
            work["stage_order"], work["estimate"], color=COLORS[key], marker=MARKERS[key],
            linestyle=LINESTYLES[key], markeredgecolor="#20242B", markeredgewidth=0.4,
            label=str(work["model"].iloc[0]),
        )
    axis.axhline(1, color="#343A40", linewidth=0.9, linestyle="--")
    axis.set_xticks(x, categories, rotation=18, ha="right")
    axis.set_ylabel("RMS relative à la pré-interface")
    axis.set_yscale("log")
    axis.legend(frameon=False)
    _clean_axes(axis)
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def plot_residual_branch_bars(data: pd.DataFrame, output_stem: Path, ylabel: str, *, zero_line: bool = False) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    figure, axis = plt.subplots(figsize=(6.9, 4.2))
    colors = ["#0072B2", "#D55E00", "#6B7280"]
    x = np.arange(len(data))
    errors = np.vstack([data["estimate"] - data["q25"], data["q75"] - data["estimate"]])
    axis.bar(x, data["estimate"], color=colors, edgecolor="#20242B", linewidth=0.5)
    axis.errorbar(x, data["estimate"], yerr=errors, fmt="none", ecolor="#20242B", elinewidth=0.8, capsize=3)
    if zero_line:
        axis.axhline(0, color="#343A40", linewidth=0.9, linestyle="--")
    axis.set_xticks(x, data["branch_fr"], rotation=16, ha="right")
    axis.set_ylabel(ylabel)
    _clean_axes(axis)
    figure.tight_layout()
    _save_triplet(figure, output_stem)


def _write_plot_data(data: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = output_dir / "figure_data" / f"{filename}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def _write_method_file(
    output_dir: Path,
    filename: str,
    *,
    concept: str,
    models: str,
    metric: str,
    source: str,
    cohort: str,
    interpretation: str,
    limitation: str,
) -> Path:
    path = output_dir / "methods" / f"{filename}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            f"# {filename}", "", f"- Concept : {concept}", f"- Modèles : {models}",
            "- Attaque : FGSM blanche propre au modèle", "- Epsilon : 0,10",
            f"- Cohorte : {cohort}", f"- Métrique : {metric}",
            "- Agrégation : médiane par graine, puis médiane des graines; bande ou barre d’intervalle = intervalle interquartile des observations appariées.",
            "- Normalisation : RMS par dimension; dommage normalisé seulement lorsque l’axe le précise, avec tau = 10^-3.",
            "- Exclusions : valeurs non finies de la métrique; aucun nouvel exemple ni nouveau checkpoint n’a été calculé.",
            f"- Source : {source}", f"- Interprétation : {interpretation}",
            f"- Limitation : {limitation}",
            "- Ne permet pas d’établir : une causalité proprement quantique, une protection causée par la segmentation, ou un effet monotone de la profondeur.",
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def _record_figure(
    records: list[dict[str, object]],
    output_dir: Path,
    filename: str,
    data: pd.DataFrame,
    *,
    concept: str,
    models: str,
    metric: str,
    recommended_use: str,
    cohort: str,
    interpretation: str,
    limitation: str,
    plotting_function: str,
) -> None:
    data_path = _write_plot_data(data, output_dir, filename)
    method_path = _write_method_file(
        output_dir, filename, concept=concept, models=models, metric=metric,
        source="; ".join(sorted(set(data.get("source_artifact", pd.Series(["trace CSV"])).dropna().astype(str)))),
        cohort=cohort, interpretation=interpretation, limitation=limitation,
    )
    records.append({
        "figure_id": filename,
        "filename": f"figures/{filename}.png",
        "pdf_filename": f"figures/{filename}.pdf",
        "svg_filename": f"figures/{filename}.svg",
        "data_csv": data_path.relative_to(output_dir).as_posix(),
        "method_file": method_path.relative_to(output_dir).as_posix(),
        "concept": concept,
        "models": models,
        "attack": "FGSM blanche propre au modèle",
        "epsilon": PRIMARY_EPSILON,
        "cohort_size": cohort,
        "metric": metric,
        "recommended_use": recommended_use,
        "status": "ready",
        "plotting_function": plotting_function,
        "generator_script": "scripts/generate_trace_figure_redesign.py",
    })


def _per_model_interpretation(data: pd.DataFrame, metric: str) -> str:
    if metric == "delta_rms":
        first = float(data.sort_values("stage_order")["estimate"].iloc[1])
        start = float(data.sort_values("stage_order")["estimate"].iloc[0])
        peak = float(data["estimate"].max())
        if first < start and peak > first:
            return "Visualise une atténuation initiale suivie d’une remontée possible; le premier bloc ne fixe donc pas à lui seul la trajectoire finale."
        if first > start:
            return "Met en évidence une amplification RMS dès le premier bloc, sans établir que cette amplification cause à elle seule l’erreur finale."
        return "Visualise la trajectoire du déplacement RMS sans mécanisme causal isolé."
    if metric == "projected_margin_harm":
        return "Visualise quand le déplacement devient aligné avec une réduction locale de la marge de décision."
    return "Visualise la taille du déplacement relativement à l’activation propre de chaque bloc."


def _write_per_model_figures(
    traces: Mapping[str, pd.DataFrame], output_dir: Path, records: list[dict[str, object]]
) -> dict[tuple[str, str], pd.DataFrame]:
    datasets = per_model_progression_data(traces)
    suffix = {
        "delta_rms": "rms_progression",
        "projected_margin_harm": "margin_harm_progression",
        "relative_displacement": "relative_displacement_progression",
    }
    label = {
        "delta_rms": "déplacement RMS",
        "projected_margin_harm": "dommage de marge projeté",
        "relative_displacement": "déplacement L2 relatif",
    }
    for view in tqdm(MODEL_VIEWS, desc="Figures par modèle"):
        for metric in ("delta_rms", "projected_margin_harm", "relative_displacement"):
            data = datasets.get((view.key, metric))
            if data is None or data.empty:
                continue
            filename = f"{view.slug}_{suffix[metric]}"
            plot_progression(
                data, output_dir / "figures" / filename, metric,
                residual_routing=(view.key == "reshvql"),
            )
            _record_figure(
                records, output_dir, filename, data,
                concept=f"Progression intra-modèle — {label[metric]}",
                models=view.label, metric=label[metric], recommended_use="annexe; candidat principal pour HVQL",
                cohort=view.evidence, interpretation=_per_model_interpretation(data, metric),
                limitation=(
                    "Diagnostic ancien, une seule graine et 19 exemples." if view.source == "legacy"
                    else "L’intervalle interquartile décrit l’hétérogénéité des exemples, pas une incertitude entre entraînements."
                ),
                plotting_function="plot_progression",
            )
    return datasets


def _write_grouped_figures(
    traces: Mapping[str, pd.DataFrame], output_dir: Path, records: list[dict[str, object]]
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    requests = (
        ("grouped_late_hybrids_rms_progression", "delta_rms", "déplacement RMS"),
        ("grouped_late_hybrids_margin_harm_progression", "normalized_margin_harm", "dommage de marge normalisé"),
        ("grouped_late_hybrids_alignment_progression", "alignment_harmful", "alignement nuisible"),
    )
    for filename, metric, metric_label in requests:
        data = late_hybrid_grouped_data(traces, metric)
        datasets[filename] = data
        plot_grouped_progression(data, output_dir / "figures" / filename, metric)
        _record_figure(
            records, output_dir, filename, data,
            concept="Comparaison des chemins décisionnels hybrides tardifs",
            models="HVQL; HPVQL (5 blocs); HVQM", metric=metric_label,
            recommended_use="section principale" if metric != "alignment_harmful" else "figure de soutien",
            cohort="500 exemples par graine; graines 0 et 42",
            interpretation="Met en évidence les différences de propagation sur des rôles architecturaux communs, tout en laissant un vide lorsque HVQM ne possède pas de bloc distinct correspondant.",
            limitation="L’alignement par rôle ne rend pas les transformations internes identiques et n’isole pas causalement la responsabilité quantique.",
            plotting_function="plot_grouped_progression",
        )
    return datasets


def _write_decision_figures(
    traces: Mapping[str, pd.DataFrame], output_dir: Path, records: list[dict[str, object]]
) -> dict[str, pd.DataFrame]:
    datasets = {}
    requests = (
        ("decision_stage_comparison_rms", "delta_rms", r"Déplacement RMS au bloc décisionnel", False),
        ("decision_stage_comparison_margin_harm", "projected_margin_harm", r"Dommage de marge projeté $H$", True),
        ("decision_stage_comparison_harm_efficiency", "harm_per_delta_l2", r"Dommage par unité de déplacement $H/\|\Delta a\|_2$", True),
    )
    for filename, metric, axis_label, zero in requests:
        data = decision_stage_data(traces, metric)
        datasets[filename] = data
        plot_horizontal_interval(data, output_dir / "figures" / filename, axis_label, zero_line=zero)
        _record_figure(
            records, output_dir, filename, data,
            concept="Responsabilité quantique et routage décisionnel",
            models="CNN; HVQM; HVQL; HVQL résiduel", metric=metric,
            recommended_use="figure de soutien",
            cohort="19 exemples communs correctement classés; graine 0",
            interpretation="Compare des représentations proches de la décision finale plutôt que des couches de même profondeur nominale.",
            limitation="Les dimensions, les gradients aval et les capacités des modèles diffèrent; la comparaison ne mesure pas une responsabilité quantique pure.",
            plotting_function="plot_horizontal_interval",
        )
    margins = logits_margin_data(traces)
    datasets["decision_stage_comparison_logits_margins"] = margins
    plot_margin_pairs(margins, output_dir / "figures" / "decision_stage_comparison_logits_margins")
    _record_figure(
        records, output_dir, "decision_stage_comparison_logits_margins", margins,
        concept="Marges propre et adversaire aux logits", models="CNN; HVQM; HVQL; HVQL résiduel",
        metric="marge de décision", recommended_use="annexe",
        cohort="19 exemples communs correctement classés; graine 0",
        interpretation="Visualise la marge exacte avant et après l’attaque sur le même petit ensemble apparié.",
        limitation="La petite cohorte de diagnostic ne doit pas être confondue avec l’exactitude robuste du jeu de test complet.",
        plotting_function="plot_margin_pairs",
    )
    return datasets


def _write_internal_variation_figures(
    traces: Mapping[str, pd.DataFrame], output_dir: Path, records: list[dict[str, object]]
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    groups = (
        ("hvql_vs_hpvql", ("hvql", "hpvql5"), "HVQL; HPVQL (5 blocs)", "section principale"),
        ("hpvql_blocks", ("hpvql5", "hpvql10"), "HPVQL (5 blocs); HPVQL (10 blocs)", "annexe"),
    )
    for prefix, models, labels, use in groups:
        for metric, suffix, metric_label in (
            ("delta_rms", "rms_progression", "déplacement RMS"),
            ("normalized_margin_harm", "margin_harm_progression", "dommage de marge normalisé"),
        ):
            data = segmentation_progression_data(traces, models, metric)
            filename = f"{prefix}_{suffix}"
            datasets[filename] = data
            plot_grouped_progression(data, output_dir / "figures" / filename, metric)
            _record_figure(
                records, output_dir, filename, data,
                concept="Variation interne du bloc quantique", models=labels, metric=metric_label,
                recommended_use=use, cohort="500 exemples par graine; graines 0 et 42",
                interpretation="Visualise des trajectoires proches malgré une organisation quantique globale ou partitionnée différente.",
                limitation="Les checkpoints ont été entraînés séparément; la segmentation n’est pas isolée causalement et la dimension mesurée diffère.",
                plotting_function="plot_grouped_progression",
            )
        for metric, suffix, metric_label in (
            ("delta_rms", "transition_rms", "déplacement RMS"),
            ("normalized_margin_harm", "transition_margin_harm", "dommage de marge normalisé"),
        ):
            full = segmentation_progression_data(traces, models, metric)
            data = full[full["layer"].isin(["prequantum", "measurement", "logits"])].copy()
            filename = f"{prefix}_{suffix}"
            datasets[filename] = data
            plot_grouped_progression(data, output_dir / "figures" / filename, metric)
            _record_figure(
                records, output_dir, filename, data,
                concept="Transition pré-interface → mesure → logits", models=labels, metric=metric_label,
                recommended_use="figure compacte de soutien", cohort="500 exemples par graine; graines 0 et 42",
                interpretation="Concentre la lecture sur la transmission à l’interface et la lecture finale.",
                limitation="La figure n’établit pas que la différence observée vient de la segmentation seule.",
                plotting_function="plot_grouped_progression",
            )
    for metric, suffix, metric_label in (
        ("delta_rms", "rms_progression", "déplacement RMS"),
        ("normalized_margin_harm", "margin_harm_progression", "dommage de marge normalisé"),
    ):
        data = legacy_variant_progression_data(traces, metric)
        filename = f"hvql_standard_residual_dropout_{suffix}"
        datasets[filename] = data
        plot_grouped_progression(data, output_dir / "figures" / filename, metric)
        _record_figure(
            records, output_dir, filename, data,
            concept="HVQL standard, résiduel et dropout", models="HVQL; HVQL résiduel; HVQL dropout",
            metric=metric_label, recommended_use="annexe",
            cohort="diagnostic de 19 exemples; graine 0",
            interpretation="Visualise des routages différents, notamment l’apparition d’une fusion résiduelle qui contourne la sortie mesurée.",
            limitation="Le petit audit à une graine ne permet pas une conclusion populationnelle; le dropout est identité en inférence.",
            plotting_function="plot_grouped_progression",
        )
        transition = data[data["stage_label"].isin(["Pré-interface", "Sortie quantique", "Fusion résiduelle", "Logits"])].copy()
        transition_filename = f"hvql_variants_transition_{'rms' if metric == 'delta_rms' else 'margin_harm'}"
        datasets[transition_filename] = transition
        plot_grouped_progression(transition, output_dir / "figures" / transition_filename, metric)
        _record_figure(
            records, output_dir, transition_filename, transition,
            concept="Transition HVQL standard / résiduel / dropout", models="HVQL; HVQL résiduel; HVQL dropout",
            metric=metric_label, recommended_use="annexe",
            cohort="diagnostic de 19 exemples; graine 0",
            interpretation="Isole visuellement les blocs proches de l’interface et de la lecture finale.",
            limitation="La cohorte et la graine uniques rendent la figure descriptive seulement.",
            plotting_function="plot_grouped_progression",
        )
    return datasets


def _residual_route_comparison(traces: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = traces["legacy"]
    rows = []
    selections = {
        "hvql": {
            "pre-quantum latent": (0, "Pré-interface"),
            "measured quantum vector": (1, "Mesure / branche q"),
            "final logits": (4, "Logits"),
        },
        "reshvql": {
            "pre-quantum latent": (0, "Pré-interface"),
            "quantum branch": (1, "Mesure / branche q"),
            "classical bypass": (2, "Bypass classique"),
            "residual merge": (3, "Fusion"),
            "final logits": (4, "Logits"),
        },
    }
    for key, mapping in selections.items():
        work = frame[(frame["model_key"] == key) & frame["layer"].isin(mapping)].copy()
        wide = work.pivot(index="dataset_index", columns="layer", values="delta_rms")
        pre = "pre-quantum latent"
        for layer, (order, label) in mapping.items():
            ratio = wide[layer] / wide[pre].replace(0, np.nan)
            rows.append({
                "model_key": key, "model": _display_label(key), "layer": layer,
                "stage_order": order, "stage_label": label,
                "estimate": float(ratio.median()), "q25": float(ratio.quantile(0.25)),
                "q75": float(ratio.quantile(0.75)), "n_samples": int(ratio.notna().sum()),
                "n_seeds": 1, "metric": "rms_relative_to_preinterface", "epsilon": PRIMARY_EPSILON,
                "source_artifact": "layerwise_trace_samples.csv",
            })
    return pd.DataFrame(rows)


def _write_interface_figures(
    artifacts: Mapping[str, pd.DataFrame], traces: Mapping[str, pd.DataFrame],
    output_dir: Path, records: list[dict[str, object]],
) -> dict[str, pd.DataFrame]:
    datasets = {}
    summary = interface_summary_data(artifacts, traces)
    datasets["interface_selectivity_transition_comparison"] = summary
    plot_interface_transition(summary, output_dir / "figures" / "interface_selectivity_transition_comparison")
    _record_figure(
        records, output_dir, "interface_selectivity_transition_comparison", summary,
        concept="Sélectivité de l’interface quantique", models="HVQL; HPVQL; HVQM; HQE",
        metric="RMS pré-interface et mesurée; dommage normalisé aux logits",
        recommended_use="meilleure figure de sélectivité",
        cohort="500 exemples par graine; graines 0 et 42",
        interpretation="Met en évidence que la variation de taille observée à la mesure et le dommage final ne sont pas la même quantité; l’annotation donne le dommage normalisé aux logits.",
        limitation="Les rapports dépendent de la dimension mesurée, du gradient aval et du petit nombre d’architectures; HQE présente une variabilité de graine importante.",
        plotting_function="plot_interface_transition",
    )
    datasets["interface_retention_vs_harm"] = summary
    plot_retention_harm(summary, output_dir / "figures" / "interface_retention_vs_harm")
    _record_figure(
        records, output_dir, "interface_retention_vs_harm", summary,
        concept="Rétention RMS et concentration du dommage", models="HVQL; HPVQL; HVQM; HQE",
        metric="rétention RMS; variation du dommage normalisé",
        recommended_use="figure principale ou soutien immédiat",
        cohort="500 exemples par graine; graines 0 et 42",
        interpretation="Visualise les cas où une faible rétention RMS coexiste avec une hausse du dommage projeté à la mesure, ce qui est compatible avec une transmission directionnelle.",
        limitation="Quatre points ne permettent ni corrélation stable ni attribution causale à un mécanisme quantique.",
        plotting_function="plot_retention_harm",
    )
    residual = _residual_route_comparison(traces)
    datasets["hvql_vs_residual_interface_comparison"] = residual
    plot_residual_interface_comparison(residual, output_dir / "figures" / "hvql_vs_residual_interface_comparison")
    _record_figure(
        records, output_dir, "hvql_vs_residual_interface_comparison", residual,
        concept="Contournement résiduel de l’interface", models="HVQL; HVQL résiduel",
        metric="RMS relative à la pré-interface", recommended_use="meilleure figure de routage résiduel",
        cohort="diagnostic de 19 exemples; graine 0",
        interpretation="Met en évidence que la branche classique transporte un déplacement autour de la sortie mesurée avant la fusion.",
        limitation="Le diagnostic ne sépare pas un effet quantique d’un effet de goulot, de mesure ou de capacité de lecture.",
        plotting_function="plot_residual_interface_comparison",
    )
    for metric, filename, ylabel, zero in (
        ("delta_rms_contribution", "residual_branch_rms_comparison", r"Contribution RMS du déplacement", False),
        ("normalized_margin_harm", "residual_branch_margin_harm_comparison", r"Dommage de marge normalisé", True),
    ):
        branch = residual_branch_data(artifacts, metric)
        datasets[filename] = branch
        plot_residual_branch_bars(branch, output_dir / "figures" / filename, ylabel, zero_line=zero)
        _record_figure(
            records, output_dir, filename, branch,
            concept="Contributions des branches du HVQL résiduel", models="HVQL résiduel",
            metric=metric, recommended_use="figure de soutien",
            cohort="500 exemples par graine; graines 0 et 42",
            interpretation="Compare directement la branche quantique, le contournement classique mis à l’échelle et leur fusion.",
            limitation="Les contributions de dommage dépendent des gradients propres à chaque branche et ne sont pas des effets causaux indépendants.",
            plotting_function="plot_residual_branch_bars",
        )
    return datasets


def _per_model_stage_mapping(traces: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for view in MODEL_VIEWS:
        raw = model_trace(traces, view)
        rows.append({
            "model": view.label,
            "model_key": view.key,
            "raw_traced_layers": "; ".join(raw.sort_values("stage_order")["layer"].drop_duplicates()),
            "grouped_stages_used_in_final_figures": "; ".join(raw.sort_values("stage_order")["stage_label"].drop_duplicates()),
            "primary_source_artifact": str(raw["evidence_source"].iloc[0]),
            "evidence_scope": view.evidence,
        })
    return pd.DataFrame(rows)


def _paired_stage_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    wide = frame[frame["layer"].isin([numerator, denominator])].pivot_table(
        index=["seed", "dataset_index"], columns="layer", values="delta_rms", aggfunc="first"
    )
    if numerator not in wide or denominator not in wide:
        return np.nan
    ratio = wide[numerator] / wide[denominator].replace(0, np.nan)
    seed_values = ratio.groupby(level="seed").median()
    return float(seed_values.median())


def _progression_summary(traces: Mapping[str, pd.DataFrame], datasets: Mapping[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for view in MODEL_VIEWS:
        raw = model_trace(traces, view)
        ordered_layers = list(view.layers)
        first_ratio = _paired_stage_ratio(raw, ordered_layers[1], ordered_layers[0])
        stage_ratios = [_paired_stage_ratio(raw, layer, ordered_layers[0]) for layer in ordered_layers[1:]]
        maximum_ratio = float(np.nanmax(stage_ratios)) if np.isfinite(stage_ratios).any() else np.nan
        rms_summary = datasets[(view.key, "delta_rms")].set_index("layer")
        harm_summary = datasets[(view.key, "projected_margin_harm")].set_index("layer")
        pre_layer = {
            "hvql": "prequantum", "hpvql5": "prequantum", "hvqm": "encoder", "hqe": "prequantum",
            "reshvql": "pre-quantum latent", "drophvql": "pre-dropout latent",
        }.get(view.key)
        measured_layer = {
            "hvql": "measurement", "hpvql5": "measurement", "hvqm": "measurement", "hqe": "measurement",
            "reshvql": "quantum branch", "drophvql": "measured quantum vector",
        }.get(view.key)
        pre_rms = float(rms_summary.loc[pre_layer, "estimate"]) if pre_layer in rms_summary.index else np.nan
        measured_rms = float(rms_summary.loc[measured_layer, "estimate"]) if measured_layer in rms_summary.index else np.nan
        final_layer = "logits" if "logits" in harm_summary.index else "final logits"
        final_harm = float(harm_summary.loc[final_layer, "estimate"]) if final_layer in harm_summary.index else np.nan
        if np.isfinite(pre_rms) and np.isfinite(measured_rms):
            relation = measured_rms / pre_rms if pre_rms else np.nan
            interpretation = (
                "La mesure réduit le RMS; la pertinence décisionnelle doit être lue séparément."
                if relation < 1 else
                "La mesure ne réduit pas le RMS dans cette cohorte; aucune atténuation uniforme n’est visible."
            )
        elif first_ratio < 1 and maximum_ratio > 1:
            interpretation = "Atténuation initiale puis réamplification à un bloc ultérieur."
        elif first_ratio >= 1:
            interpretation = "Amplification RMS dès le premier bloc; le dommage final reste une quantité distincte."
        else:
            interpretation = "Trajectoire descriptive; aucun mécanisme causal isolé."
        rows.append({
            "model": view.label,
            "model_key": view.key,
            "first_block_rms_change": first_ratio - 1 if np.isfinite(first_ratio) else np.nan,
            "first_block_rms_ratio": first_ratio,
            "maximum_rms_change": maximum_ratio - 1 if np.isfinite(maximum_ratio) else np.nan,
            "maximum_rms_ratio_to_input": maximum_ratio,
            "pre_interface_rms": pre_rms,
            "measured_rms": measured_rms,
            "final_margin_harm": final_harm,
            "recommended_interpretation": interpretation,
            "evidence_scope": view.evidence,
            "source_artifact": str(raw["evidence_source"].iloc[0]),
        })
    return pd.DataFrame(rows)


def _recommendations(records: pd.DataFrame) -> pd.DataFrame:
    shortlist = {
        "hvql_d50_rms_progression": (
            "Progression intra-modèle HVQL", "Montre des blocs réels, le passage latent→mesure et la lecture finale.",
            "Ne compare pas les architectures et le RMS seul ne mesure pas le dommage décisionnel.",
            "meilleure figure pour expliquer la progression",
        ),
        "interface_retention_vs_harm": (
            "Sélectivité de l’interface", "Sépare explicitement rétention de norme et changement de dommage.",
            "Seulement quatre architectures; aucune relation causale ne peut être estimée.",
            "meilleure figure pour l’interface sélective",
        ),
        "grouped_late_hybrids_margin_harm_progression": (
            "Différences architecturales", "Aligne HVQL, HPVQL et HVQM par rôle décisionnel et conserve le signe du dommage.",
            "HVQM ne possède pas tous les rôles distincts et la normalisation dépend du dommage d’entrée.",
            "meilleure figure de soutien architectural",
        ),
    }
    rows = []
    for filename, (concept, strengths, weaknesses, use) in shortlist.items():
        row = records[records["figure_id"] == filename]
        if row.empty:
            continue
        rows.append({
            "figure_filename": str(row["filename"].iloc[0]),
            "concept": concept,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommended_use": use,
        })
    return pd.DataFrame(rows)


def _unavailable_requests() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "requested_group": "HVQL strong / basic / random",
            "status": "non récupérable depuis les traces sauvegardées",
            "reason": "Les métriques d’attaque existent par variante, mais aucun tenseur ou résumé couche par couche n’a été exporté pour basic/random.",
            "needed_evidence": "Traçage couche par couche des mêmes exemples et graines pour les trois ansatz.",
        },
        {
            "requested_group": "HVQL profondeur 50 / 100 / 200",
            "status": "non récupérable depuis les traces sauvegardées",
            "reason": "Les profondeurs 100/200 disponibles dans le vieux traçage concernent VQM, pas HVQL.",
            "needed_evidence": "Traçage des checkpoints HVQL strong qdepth=50, 100 et 200 sur une cohorte appariée.",
        },
        {
            "requested_group": "MLP [1024,512,64,64] — déplacement relatif",
            "status": "non récupérable depuis les traces sauvegardées",
            "reason": "Le nouveau CSV conserve delta_rms mais pas la norme de l’activation propre; le rapport relatif ne peut pas être dérivé.",
            "needed_evidence": "Norme L2 ou RMS des activations propres par couche pour le grand MLP.",
        },
    ])


def _caption_for_record(row: pd.Series) -> str:
    filename = Path(str(row["filename"])).name
    concept = str(row["concept"])
    return (
        rf"\texttt{{{filename}}} — {concept}. Les points représentent les médianes; "
        rf"les intervalles indiquent l’intervalle interquartile. Attaque FGSM blanche propre au modèle, $\varepsilon=0{{,}}10$."
    )


def _write_supporting_files(
    artifacts: Mapping[str, pd.DataFrame], traces: Mapping[str, pd.DataFrame],
    output_dir: Path, records: list[dict[str, object]],
    per_model_datasets: Mapping[tuple[str, str], pd.DataFrame],
    grouped_datasets: Mapping[str, pd.DataFrame],
) -> None:
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    _per_model_stage_mapping(traces).to_csv(output_dir / "per_model_stage_mapping.csv", index=False)
    _progression_summary(traces, per_model_datasets).to_csv(output_dir / "per_model_progression_summary.csv", index=False)
    grouped = []
    for name, data in grouped_datasets.items():
        if data.empty:
            continue
        copy = data.copy()
        copy.insert(0, "comparison_figure", name)
        grouped.append(copy)
    pd.concat(grouped, ignore_index=True).to_csv(output_dir / "grouped_comparison_summary.csv", index=False)
    _recommendations(manifest).to_csv(output_dir / "figure_recommendation_summary.csv", index=False)
    _unavailable_requests().to_csv(output_dir / "unavailable_requested_figures.csv", index=False)
    discover_trace_artifacts().to_csv(output_dir / "source_inventory.csv", index=False)
    captions = ["% Légendes proposées — paquet de figures redessiné", ""]
    for _, row in manifest.iterrows():
        captions.append(f"% {row['figure_id']}")
        captions.append(_caption_for_record(row))
        captions.append("")
    (output_dir / "captions_fr.tex").write_text("\n".join(captions), encoding="utf-8")
    chart_map = manifest[[
        "figure_id", "concept", "models", "metric", "filename", "data_csv",
        "recommended_use", "plotting_function", "generator_script",
    ]]
    chart_map.to_csv(output_dir / "chart_map.csv", index=False)
    (output_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (output_dir / "notebooks").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT_ROOT / "src" / "analytics" / "attacks" / "trace_figure_redesign.py",
        output_dir / "scripts" / "trace_figure_redesign.py",
    )
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "generate_trace_figure_redesign.py",
        output_dir / "scripts" / "generate_trace_figure_redesign.py",
    )
    shutil.copy2(
        PROJECT_ROOT / "src" / "analytics" / "attacks" / "trace_figure_redesign.ipynb",
        output_dir / "notebooks" / "trace_figure_redesign.ipynb",
    )
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(output_dir).as_posix(),
                "suffix": path.suffix.lower(),
                "bytes": path.stat().st_size,
            })
    pd.DataFrame(files).to_csv(output_dir / "output_inventory.csv", index=False)


def build_redesigned_figure_package(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    """Generate the complete figure package from saved CSV traces only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_trace_artifacts()
    traces = prepared_traces(artifacts)
    records: list[dict[str, object]] = []

    per_model = _write_per_model_figures(traces, output_dir, records)
    grouped_late = _write_grouped_figures(traces, output_dir, records)
    decision = _write_decision_figures(traces, output_dir, records)
    internal = _write_internal_variation_figures(traces, output_dir, records)
    interface = _write_interface_figures(artifacts, traces, output_dir, records)

    grouped_all = {**grouped_late, **decision, **internal, **interface}
    _write_supporting_files(artifacts, traces, output_dir, records, per_model, grouped_all)
    result = {
        "output_dir": str(output_dir),
        "figure_count": len(records),
        "png_count": len(list((output_dir / "figures").glob("*.png"))),
        "pdf_count": len(list((output_dir / "figures").glob("*.pdf"))),
        "svg_count": len(list((output_dir / "figures").glob("*.svg"))),
        "unavailable_groups": len(_unavailable_requests()),
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
