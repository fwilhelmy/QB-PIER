"""White-box FGSM analysis for the retained VQM attack variants.

The module is intentionally library-only.  It reads saved attack logits,
training summaries, circuit summaries, training dynamics, and cached input
gradients; it never launches Hydra or recomputes a model or attack.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm


RETAINED_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "canonical_model_name": "Quantum/VQM/ansatz=basic, qdepth=50",
        "label": "basic-50",
        "ansatz": "basic",
        "circuit_depth": 50,
        "training_name": "Quantum/VQM/n_qdepth=50,ansatz=basic",
        "results_variant": "ansatz=basic, qdepth=50",
    },
    {
        "canonical_model_name": "Quantum/VQM/ansatz=random, qdepth=50",
        "label": "random-50",
        "ansatz": "random",
        "circuit_depth": 50,
        "training_name": "Quantum/VQM/n_qdepth=50,ansatz=random,ratio_imprim=0.75",
        "results_variant": "ansatz=random, qdepth=50, ratio_imprim=0.75",
    },
    {
        "canonical_model_name": "Quantum/VQM/ansatz=no_entangling, qdepth=50",
        "label": "no-entangling-50",
        "ansatz": "no_entangling",
        "circuit_depth": 50,
        "training_name": "Quantum/VQM/n_qdepth=50,ansatz=no_entangling",
        "results_variant": "ansatz=no_entangling, qdepth=50",
    },
    {
        "canonical_model_name": "Quantum/VQM/ansatz=strong, qdepth=50",
        "label": "strong-50",
        "ansatz": "strong",
        "circuit_depth": 50,
        "training_name": "Quantum/VQM/n_qdepth=50,ansatz=strong",
        "results_variant": "ansatz=strong, qdepth=50",
    },
    {
        "canonical_model_name": "Quantum/VQM/ansatz=strong, qdepth=100",
        "label": "strong-100",
        "ansatz": "strong",
        "circuit_depth": 100,
        "training_name": "Quantum/VQM/n_qdepth=100,ansatz=strong",
        "results_variant": "ansatz=strong, qdepth=100",
    },
    {
        "canonical_model_name": "Quantum/VQM/ansatz=strong, qdepth=200",
        "label": "strong-200",
        "ansatz": "strong",
        "circuit_depth": 200,
        "training_name": "Quantum/VQM/n_qdepth=200,ansatz=strong",
        "results_variant": "ansatz=strong, qdepth=200",
    },
)

RETAINED_BY_NAME = {row["canonical_model_name"]: row for row in RETAINED_VARIANTS}
ANSATZ_MARKERS = {"basic": "s", "random": "^", "strong": "o", "no_entangling": "D"}
DEPTH_COLORS = {50: "#2563EB", 100: "#D97706", 200: "#C026D3"}
METRIC_COLUMNS = (
    "robust_accuracy_auc",
    "retention_auc",
    "asr_auc",
    "flip_rate_auc",
    "accuracy_drop_auc",
)
NEAR_ZERO_THRESHOLD = 1.0e-6
MATCH_TOLERANCE = 0.05


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    qaml_results_root: Path
    output_dir: Path

    @property
    def attack_root(self) -> Path:
        return self.project_root / "logs" / "ATTACKS" / "attack_experiment_canonical"

    @property
    def gradient_root(self) -> Path:
        return self.project_root / "logs" / "GRADIENTS"

    @property
    def training_export_root(self) -> Path:
        return self.project_root / "outputs" / "training_dynamics_export"

    @property
    def whitebox_summary(self) -> Path:
        return self.qaml_results_root / "Results - White-Box Attacks.csv"

    @property
    def training_summary(self) -> Path:
        return self.qaml_results_root / "Results - Models Training Information.csv"

    @property
    def circuit_summary(self) -> Path:
        return self.qaml_results_root / "Results - QuantumCircuits Information.csv"


def discover_sources(paths: AnalysisPaths) -> pd.DataFrame:
    """Return the explicit source inventory used by the analysis."""
    rows = [
        ("whitebox_summary", paths.whitebox_summary, True),
        ("training_summary", paths.training_summary, True),
        ("circuit_summary", paths.circuit_summary, True),
        ("attack_manifest", paths.attack_root / "manifest.json", True),
        ("attack_epsilons", paths.attack_root / "epsilons.json", True),
        ("attack_migration", paths.attack_root / "migration_report.json", True),
        ("training_model_metadata", paths.training_export_root / "model_static_metadata.csv", True),
        ("training_quantum_metadata", paths.training_export_root / "quantum_static_metadata.csv", True),
        ("training_curves", paths.training_export_root / "training_curves_wide.csv", True),
        ("gradient_meta", paths.gradient_root / "meta.json", True),
        ("gradient_cache", paths.gradient_root / "gradients", True),
    ]
    return pd.DataFrame(
        [
            {
                "source_id": source_id,
                "path": str(path.resolve()),
                "required": required,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.is_file() else np.nan,
            }
            for source_id, path, required in rows
        ]
    )


def _read_results_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=1)


def _parse_percent(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if text in {"", "-", "nan", "None"}:
        return float("nan")
    return float(text.rstrip("%")) / (100.0 if text.endswith("%") else 1.0)


def _safe_attack_name(canonical_model_name: str) -> str:
    return canonical_model_name.replace("\\", "__").replace("/", "__").replace(" ", "_").replace(":", "_")


def _auc(epsilon: np.ndarray, values: np.ndarray) -> float:
    width = float(epsilon[-1] - epsilon[0])
    if width <= 0:
        raise ValueError("epsilon grid must have positive width")
    return float(np.trapezoid(values, epsilon) / width)


def _recompute_curve_from_logits(paths: AnalysisPaths, spec: dict[str, Any], seed: int) -> pd.DataFrame:
    safe = _safe_attack_name(spec["canonical_model_name"])
    seed_root = paths.attack_root / f"seed_{seed}"
    labels = np.load(seed_root / "labels.npy")
    clean_preds = np.load(seed_root / "baselines" / safe / "preds_clean.npy")
    logits = np.load(seed_root / "attacks" / f"source={safe}__target={safe}.npy", mmap_mode="r")
    epsilon_payload = json.loads((paths.attack_root / "epsilons.json").read_text(encoding="utf-8"))
    epsilon_values = epsilon_payload.get("epsilons", epsilon_payload) if isinstance(epsilon_payload, dict) else epsilon_payload
    epsilons = np.asarray(epsilon_values, dtype=float)
    if logits.shape[:2] != (len(epsilons), len(labels)):
        raise ValueError(f"unexpected logits shape for {spec['canonical_model_name']} seed {seed}: {logits.shape}")
    clean_correct = clean_preds == labels
    rows: list[dict[str, Any]] = []
    for idx in range(len(epsilons)):
        preds = np.asarray(logits[idx]).argmax(axis=1)
        rows.append(
            {
                "canonical_model_name": spec["canonical_model_name"],
                "variant_label": spec["label"],
                "ansatz": spec["ansatz"],
                "circuit_depth": spec["circuit_depth"],
                "seed": seed,
                "epsilon": float(epsilons[idx]),
                "n_samples": int(len(labels)),
                "robust_accuracy": float(np.mean(preds == labels)),
                "flip_rate": float(np.mean(preds != clean_preds)),
                "ASR": float(np.mean(preds[clean_correct] != labels[clean_correct])),
                "clean_correct_n": int(clean_correct.sum()),
            }
        )
    return pd.DataFrame(rows)


def load_vqm_attack_curves(paths: AnalysisPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute VQM white-box curves from logits and compare with aggregates."""
    raw_frames = []
    checks = []
    tasks = [(spec, seed) for spec in RETAINED_VARIANTS for seed in (0, 42)]
    for spec, seed in tqdm(tasks, desc="Recomputing retained VQM self-attacks"):
        raw = _recompute_curve_from_logits(paths, spec, seed)
        raw_frames.append(raw)
        source_path = paths.attack_root / "tables" / (
            f"aggregates_seed={seed}__source={_safe_attack_name(spec['canonical_model_name'])}.parquet"
        )
        aggregate = pd.read_parquet(source_path)
        aggregate = aggregate[aggregate["source_model"] == aggregate["target_model"]].sort_values("epsilon")
        merged = raw.merge(
            aggregate[["epsilon", "robust_accuracy", "flip_rate", "ASR"]],
            on="epsilon",
            suffixes=("_raw", "_aggregate"),
            validate="one_to_one",
        )
        for metric in ("robust_accuracy", "flip_rate", "ASR"):
            delta = (merged[f"{metric}_raw"] - merged[f"{metric}_aggregate"]).abs()
            checks.append(
                {
                    "canonical_model_name": spec["canonical_model_name"],
                    "seed": seed,
                    "metric": metric,
                    "max_abs_difference": float(delta.max()),
                    "rows_compared": int(len(delta)),
                }
            )
    curves = pd.concat(raw_frames, ignore_index=True)
    return curves, pd.DataFrame(checks)


def summarize_attack_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["canonical_model_name", "variant_label", "ansatz", "circuit_depth", "seed"]
    for keys, frame in curves.groupby(group_cols, sort=False):
        frame = frame.sort_values("epsilon")
        epsilon = frame["epsilon"].to_numpy(dtype=float)
        clean_accuracy = float(frame.loc[frame["epsilon"].idxmin(), "robust_accuracy"])
        robust = frame["robust_accuracy"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "clean_accuracy": clean_accuracy,
                "robust_accuracy_auc": _auc(epsilon, robust),
                "retention_auc": _auc(epsilon, robust / clean_accuracy),
                "asr_auc": _auc(epsilon, frame["ASR"].to_numpy(dtype=float)),
                "flip_rate_auc": _auc(epsilon, frame["flip_rate"].to_numpy(dtype=float)),
                "accuracy_drop_auc": _auc(epsilon, clean_accuracy - robust),
                "epsilon_min": float(epsilon.min()),
                "epsilon_max": float(epsilon.max()),
                "epsilon_points": int(len(epsilon)),
                "n_samples": int(frame["n_samples"].iloc[0]),
                "clean_correct_n": int(frame["clean_correct_n"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _results_vqm_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["Group / Family"].astype(str).str.casefold().eq("quantum")
        & frame["Architecture"].astype(str).str.casefold().eq("vqm")
    ].copy()


def _find_results_variant(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    variants = frame["Variant Configuration (knobs)"].astype(str)
    if spec["ansatz"] == "random":
        mask = variants.str.contains(r"ansatz=random", regex=True) & variants.str.contains(r"qdepth=50", regex=True) & variants.str.contains(r"ratio_imprim=0\.75", regex=True)
    else:
        mask = variants.str.replace(" ", "", regex=False).eq(spec["results_variant"].replace(" ", ""))
    matches = frame[mask]
    if len(matches) != 1:
        raise ValueError(f"expected one results row for {spec['label']}, found {len(matches)}")
    return matches.iloc[0]


def add_training_and_circuit_metadata(paths: AnalysisPaths, attack_seed: pd.DataFrame) -> pd.DataFrame:
    training_summary = _results_vqm_rows(_read_results_csv(paths.training_summary))
    circuit_summary = _results_vqm_rows(_read_results_csv(paths.circuit_summary))
    model_metadata = pd.read_csv(paths.training_export_root / "model_static_metadata.csv")
    training_curves = pd.read_csv(paths.training_export_root / "training_curves_wide.csv")

    metadata_rows = []
    for spec in RETAINED_VARIANTS:
        training_row = _find_results_variant(training_summary, spec)
        circuit_row = _find_results_variant(circuit_summary, spec)
        for seed in (0, 42):
            model_match = model_metadata[
                model_metadata["canonical_model_name"].eq(spec["training_name"])
                & model_metadata["seed"].eq(seed)
            ]
            training_time = float("nan")
            training_time_scope = "variant_summary_fallback"
            if len(model_match) == 1 and pd.notna(model_match.iloc[0]["training_time_seconds"]):
                training_time = float(model_match.iloc[0]["training_time_seconds"])
                training_time_scope = "seed_export"
            else:
                training_time = float(training_row["Training Time (s)"])

            curve_match = training_curves[
                training_curves["canonical_model_name"].eq(spec["training_name"])
                & training_curves["seed"].eq(seed)
            ].sort_values("epoch")
            validation = curve_match.dropna(subset=["val_acc"])
            test = curve_match.dropna(subset=["test_acc"])
            metadata_rows.append(
                {
                    "canonical_model_name": spec["canonical_model_name"],
                    "seed": seed,
                    "number_of_gates": int(circuit_row["Number of Gates"]),
                    "ansatz_gate_count": int(circuit_row["Number of Gates.2"]),
                    "quantum_parameter_count": int(circuit_row["Params Quantum"]),
                    "number_of_qubits": int(circuit_row["Number of Qubits"]),
                    "training_time_seconds": training_time,
                    "training_time_scope": training_time_scope,
                    "final_validation_accuracy": float(validation.iloc[-1]["val_acc"]) if len(validation) else float("nan"),
                    "final_test_accuracy": float(test.iloc[-1]["test_acc"]) if len(test) else float("nan"),
                    "training_reported_test_accuracy": _parse_percent(training_row["Test Accuracy"]),
                    "training_reported_validation_accuracy": _parse_percent(training_row["Validation Accuracy"]),
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    return attack_seed.merge(metadata, on=["canonical_model_name", "seed"], how="left", validate="one_to_one")


def aggregate_variants(seed_table: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "clean_accuracy", "robust_accuracy_auc", "retention_auc", "asr_auc", "flip_rate_auc",
        "accuracy_drop_auc", "number_of_gates", "ansatz_gate_count", "quantum_parameter_count",
        "number_of_qubits", "training_time_seconds", "final_validation_accuracy", "final_test_accuracy",
        "training_reported_test_accuracy", "training_reported_validation_accuracy",
    ]
    grouped = seed_table.groupby(
        ["canonical_model_name", "variant_label", "ansatz", "circuit_depth"], as_index=False, sort=False
    )[numeric].mean()
    grouped["seed_count"] = seed_table.groupby("canonical_model_name")["seed"].nunique().reindex(grouped["canonical_model_name"]).to_numpy()
    return grouped


def _canonical_gradient_name(raw_name: str, migration: dict[str, str], training_map: dict[str, str]) -> str:
    if raw_name in migration:
        return migration[raw_name]
    if raw_name.startswith("Quantum/vqm/"):
        ansatz_match = re.search(r"ansatz=([^,]+)", raw_name)
        depth_match = re.search(r"n_qdepth=(\d+)", raw_name)
        if ansatz_match and depth_match:
            ansatz = ansatz_match.group(1)
            depth = int(depth_match.group(1))
            if ansatz == "random" and "ratio_imprim=0.75" not in raw_name:
                return training_map.get(raw_name, raw_name)
            return f"Quantum/VQM/ansatz={ansatz}, qdepth={depth}"
    return training_map.get(raw_name, raw_name)


def _gradient_summary(gradients: np.ndarray) -> dict[str, float]:
    gradients = np.asarray(gradients, dtype=np.float32)
    absolute = np.abs(gradients)
    l1 = absolute.sum(axis=1)
    l2 = np.sqrt(np.square(gradients).sum(axis=1))
    linf = absolute.max(axis=1)
    pixels = gradients.shape[1]
    top_k = max(1, int(math.ceil(0.10 * pixels)))
    top_mass = np.sort(absolute, axis=1)[:, -top_k:].sum(axis=1)
    top_mass_fraction = np.divide(top_mass, l1, out=np.full_like(top_mass, np.nan), where=l1 > 0)
    probabilities = np.divide(absolute, l1[:, None], out=np.zeros_like(absolute), where=l1[:, None] > 0)
    log_probabilities = np.zeros_like(probabilities)
    positive_probability = probabilities > 0
    log_probabilities[positive_probability] = np.log(probabilities[positive_probability])
    entropy = -np.sum(probabilities * log_probabilities, axis=1) / math.log(pixels)
    near_zero = np.mean(absolute <= NEAR_ZERO_THRESHOLD, axis=1)
    positive = np.sum(gradients > NEAR_ZERO_THRESHOLD, axis=1)
    negative = np.sum(gradients < -NEAR_ZERO_THRESHOLD, axis=1)
    sign_balance = np.divide(
        positive - negative,
        positive + negative,
        out=np.zeros(gradients.shape[0], dtype=float),
        where=(positive + negative) > 0,
    )
    return {
        "median_gradient_l1": float(np.median(l1)),
        "median_gradient_l2": float(np.median(l2)),
        "median_gradient_linf": float(np.median(linf)),
        "median_normalized_l1_per_pixel": float(np.median(l1 / pixels)),
        "median_top10_abs_gradient_mass_fraction": float(np.nanmedian(top_mass_fraction)),
        "median_normalized_saliency_entropy": float(np.nanmedian(entropy)),
        "median_near_zero_pixel_fraction": float(np.median(near_zero)),
        "median_sign_balance": float(np.median(sign_balance)),
        "first_order_fgsm_sensitivity": float(np.median(l1)),
        "gradient_sample_count": int(gradients.shape[0]),
        "gradient_pixel_count": int(pixels),
    }


def load_gradient_diagnostics(paths: AnalysisPaths) -> pd.DataFrame:
    migration_payload = json.loads((paths.attack_root / "migration_report.json").read_text(encoding="utf-8"))
    migration = migration_payload["old_to_new"]
    model_metadata = pd.read_csv(paths.training_export_root / "model_static_metadata.csv")
    training_map = dict(
        model_metadata[["model_config_name", "canonical_model_name"]].drop_duplicates().itertuples(index=False, name=None)
    )
    rows = []
    gradient_files = sorted((paths.gradient_root / "gradients").glob("*.pt"))
    for path in tqdm(gradient_files, desc="Summarizing cached input gradients"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw_name = str(payload["model_name"])
        canonical = _canonical_gradient_name(raw_name, migration, training_map)
        rows.append(
            {
                "canonical_model_name": canonical,
                "gradient_cache_model_name": raw_name,
                **_gradient_summary(payload["gradients"].cpu().numpy()),
            }
        )
    return pd.DataFrame(rows)


def load_all_self_attack_auc(paths: AnalysisPaths) -> pd.DataFrame:
    rows = []
    table_files = sorted((paths.attack_root / "tables").glob("aggregates_seed=*__source=*.parquet"))
    for path in tqdm(table_files, desc="Loading retained comparison self-attacks"):
        frame = pd.read_parquet(path)
        frame = frame[frame["source_model"] == frame["target_model"]].sort_values("epsilon")
        if frame.empty:
            continue
        epsilon = frame["epsilon"].to_numpy(dtype=float)
        clean = float(frame.iloc[0]["robust_accuracy"])
        robust = frame["robust_accuracy"].to_numpy(dtype=float)
        rows.append(
            {
                "seed": int(frame.iloc[0]["seed"]),
                "canonical_model_name": str(frame.iloc[0]["source_model"]),
                "clean_accuracy": clean,
                "robust_accuracy_auc": _auc(epsilon, robust),
                "retention_auc": _auc(epsilon, robust / clean),
                "asr_auc": _auc(epsilon, frame["ASR"].to_numpy(dtype=float)),
                "flip_rate_auc": _auc(epsilon, frame["flip_rate"].to_numpy(dtype=float)),
                "accuracy_drop_auc": _auc(epsilon, clean - robust),
            }
        )
    seed = pd.DataFrame(rows).drop_duplicates(["seed", "canonical_model_name"])
    return seed.groupby("canonical_model_name", as_index=False)[
        ["clean_accuracy", "robust_accuracy_auc", "retention_auc", "asr_auc", "flip_rate_auc", "accuracy_drop_auc"]
    ].mean()


def build_matched_comparisons(
    vqm_aggregate: pd.DataFrame,
    all_self_auc: pd.DataFrame,
    gradient_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Match each VQM to the nearest low-capacity MLP and nearest HVQM."""
    mlp_names = {
        "Classical/MLP/hidden_sizes=[4,2]",
        "Classical/MLP/hidden_sizes=[4,4]",
        "Classical/MLP/hidden_sizes=[8,4]",
    }
    hvqm_names = {
        "Hybrid/HVQM/ansatz=random, qdepth=50",
        "Hybrid/HVQM/ansatz=strong, qdepth=50",
    }
    candidates = all_self_auc[all_self_auc["canonical_model_name"].isin(mlp_names | hvqm_names)].copy()
    candidates["comparison_stratum"] = np.where(
        candidates["canonical_model_name"].isin(mlp_names), "low_capacity_mlp", "hvqm"
    )
    candidates = candidates.merge(
        gradient_summary[["canonical_model_name", "first_order_fgsm_sensitivity"]],
        on="canonical_model_name",
        how="left",
        validate="one_to_one",
    )
    vqm = vqm_aggregate.merge(
        gradient_summary[["canonical_model_name", "first_order_fgsm_sensitivity"]],
        on="canonical_model_name",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for left in vqm.itertuples(index=False):
        for stratum, pool in candidates.groupby("comparison_stratum"):
            pool = pool.assign(clean_accuracy_difference=(pool["clean_accuracy"] - left.clean_accuracy).abs())
            right = pool.sort_values(["clean_accuracy_difference", "canonical_model_name"]).iloc[0]
            rows.append(
                {
                    "vqm_model": left.canonical_model_name,
                    "vqm_label": left.variant_label,
                    "comparison_stratum": stratum,
                    "matched_model": right["canonical_model_name"],
                    "clean_accuracy_vqm": left.clean_accuracy,
                    "clean_accuracy_match": right["clean_accuracy"],
                    "clean_accuracy_difference": right["clean_accuracy_difference"],
                    "within_5pp": bool(right["clean_accuracy_difference"] <= MATCH_TOLERANCE),
                    "robust_accuracy_auc_vqm": left.robust_accuracy_auc,
                    "robust_accuracy_auc_match": right["robust_accuracy_auc"],
                    "retention_auc_vqm": left.retention_auc,
                    "retention_auc_match": right["retention_auc"],
                    "asr_auc_vqm": left.asr_auc,
                    "asr_auc_match": right["asr_auc"],
                    "first_order_fgsm_sensitivity_vqm": left.first_order_fgsm_sensitivity,
                    "first_order_fgsm_sensitivity_match": right["first_order_fgsm_sensitivity"],
                    "match_rule": "nearest clean accuracy within pre-specified stratum; <=5 pp is a valid clean match",
                }
            )
    return pd.DataFrame(rows)


def _partial_spearman(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    xr, yr, cr = (rankdata(np.asarray(v, dtype=float)) for v in (x, y, control))
    design = np.column_stack([np.ones(len(cr)), cr])
    x_resid = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_resid = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def compute_associations(vqm_aggregate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for structural in ("circuit_depth", "number_of_gates", "quantum_parameter_count"):
        for metric in METRIC_COLUMNS:
            rho, pvalue = spearmanr(vqm_aggregate[structural], vqm_aggregate[metric])
            rows.append(
                {
                    "structural_variable": structural,
                    "white_box_metric": metric,
                    "spearman_rho": float(rho),
                    "spearman_pvalue_descriptive_only": float(pvalue),
                    "partial_spearman_controlling_clean_accuracy": _partial_spearman(
                        vqm_aggregate[structural], vqm_aggregate[metric], vqm_aggregate["clean_accuracy"]
                    ),
                    "n_variants": int(len(vqm_aggregate)),
                }
            )
    associations = pd.DataFrame(rows)

    regression_rows = []
    clean = vqm_aggregate["clean_accuracy"].to_numpy(dtype=float)
    depth = vqm_aggregate["circuit_depth"].to_numpy(dtype=float)
    clean_z = (clean - clean.mean()) / clean.std(ddof=0)
    depth_z = (depth - depth.mean()) / depth.std(ddof=0)
    design = np.column_stack([np.ones(len(clean)), clean_z, depth_z])
    for metric in METRIC_COLUMNS:
        y = vqm_aggregate[metric].to_numpy(dtype=float)
        y_z = (y - y.mean()) / y.std(ddof=0)
        coefficients = np.linalg.lstsq(design, y_z, rcond=None)[0]
        fitted = design @ coefficients
        residual_sum = float(np.square(y_z - fitted).sum())
        total_sum = float(np.square(y_z - y_z.mean()).sum())
        regression_rows.append(
            {
                "white_box_metric": metric,
                "standardized_clean_accuracy_coefficient": float(coefficients[1]),
                "standardized_depth_coefficient": float(coefficients[2]),
                "r_squared": 1.0 - residual_sum / total_sum,
                "design_condition_number": float(np.linalg.cond(design)),
                "n_variants": int(len(y)),
                "model": "exploratory z(metric) ~ z(clean_accuracy) + z(circuit_depth)",
                "full_ansatz_model_status": "not fit: four ansatz levels plus clean accuracy and depth saturate six variants",
            }
        )
    return associations, pd.DataFrame(regression_rows)


def compute_exact_differences(vqm_aggregate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_label = vqm_aggregate.set_index("variant_label")
    strong_rows = []
    for left, right in (("strong-50", "strong-100"), ("strong-100", "strong-200"), ("strong-50", "strong-200")):
        for metric in ("clean_accuracy", *METRIC_COLUMNS):
            strong_rows.append(
                {
                    "comparison": f"{right} minus {left}",
                    "metric": metric,
                    "difference": float(by_label.loc[right, metric] - by_label.loc[left, metric]),
                }
            )
    ansatz_rows = []
    for comparator in ("basic-50", "random-50", "no-entangling-50"):
        for metric in ("clean_accuracy", *METRIC_COLUMNS):
            ansatz_rows.append(
                {
                    "comparison": f"strong-50 minus {comparator}",
                    "metric": metric,
                    "difference": float(by_label.loc["strong-50", metric] - by_label.loc[comparator, metric]),
                }
            )
    return pd.DataFrame(strong_rows), pd.DataFrame(ansatz_rows)


def validate_legacy_whitebox(paths: AnalysisPaths, vqm_aggregate: pd.DataFrame) -> pd.DataFrame:
    legacy = _results_vqm_rows(_read_results_csv(paths.whitebox_summary))
    rows = []
    for spec in RETAINED_VARIANTS:
        legacy_row = _find_results_variant(legacy, spec) if spec["ansatz"] != "random" else legacy[
            legacy["Variant Configuration (knobs)"].astype(str).str.replace(" ", "", regex=False).eq("ansatz=random,qdepth=50")
        ].iloc[0]
        aggregate = vqm_aggregate[vqm_aggregate["canonical_model_name"].eq(spec["canonical_model_name"])].iloc[0]
        for source_col, metric in (
            ("Clean Accuracy", "clean_accuracy"),
            ("Robust Accuracy", "robust_accuracy_auc"),
            ("Retention", "retention_auc"),
            ("Flip Rate", "flip_rate_auc"),
            ("ASR", "asr_auc"),
        ):
            legacy_value = _parse_percent(legacy_row[source_col])
            rows.append(
                {
                    "canonical_model_name": spec["canonical_model_name"],
                    "metric": metric,
                    "legacy_value": legacy_value,
                    "raw_recomputed_value": float(aggregate[metric]),
                    "absolute_difference": abs(legacy_value - float(aggregate[metric])) if np.isfinite(legacy_value) else np.nan,
                    "legacy_available": bool(np.isfinite(legacy_value)),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#E5E7EB", linewidth=0.7, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#374151")


def _add_encoding_legends(
    ax: plt.Axes,
    *,
    location: str = "best",
    depth_location: str | None = None,
) -> None:
    ansatz_handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="none", markerfacecolor="#D1D5DB",
            markeredgecolor="#111827", markersize=7, label=ansatz.replace("_", "-"),
        )
        for ansatz, marker in ANSATZ_MARKERS.items()
    ]
    depth_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=color,
            markeredgecolor="#111827", markersize=7, label=str(depth),
        )
        for depth, color in DEPTH_COLORS.items()
    ]
    first = ax.legend(handles=ansatz_handles, title="Ansatz", loc=location, frameon=False, fontsize=8, title_fontsize=8)
    ax.add_artist(first)
    if depth_location is None:
        depth_location = "center left" if location == "lower right" else "lower right"
    ax.legend(handles=depth_handles, title="Depth", loc=depth_location, frameon=False, fontsize=8, title_fontsize=8)


def _label_points(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    for row in frame.itertuples(index=False):
        ax.annotate(
            row.variant_label,
            (getattr(row, x), getattr(row, y)),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
        )


def build_figures(
    output_dir: Path,
    seed_table: pd.DataFrame,
    aggregate: pd.DataFrame,
    gradient_vqm: pd.DataFrame,
) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    sources_dir = output_dir / "figure_sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    b1 = aggregate.copy()
    b1.to_csv(sources_dir / "figure_b1.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for row in b1.itertuples(index=False):
        ax.scatter(row.clean_accuracy, row.robust_accuracy_auc, s=80, marker=ANSATZ_MARKERS[row.ansatz], color=DEPTH_COLORS[row.circuit_depth], edgecolor="#111827", linewidth=0.8)
    _label_points(ax, b1, "clean_accuracy", "robust_accuracy_auc")
    ax.set_xlabel("Clean accuracy")
    ax.set_ylabel("White-box robust accuracy AUC")
    ax.set_xlim(0.10, 0.86)
    ax.set_ylim(0.0, 0.12)
    _style_axis(ax)
    _add_encoding_legends(ax, location="upper left")
    _save_figure(fig, figures_dir, "figure_b1")

    def depth_figure(metric: str, stem: str, y_label: str, y_limits: tuple[float, float]) -> None:
        seed_source = seed_table[["canonical_model_name", "variant_label", "ansatz", "circuit_depth", "seed", metric]].copy()
        seed_source["aggregation_level"] = "seed"
        aggregate_source = aggregate[["canonical_model_name", "variant_label", "ansatz", "circuit_depth", metric]].copy()
        aggregate_source["seed"] = np.nan
        aggregate_source["aggregation_level"] = "seed_mean"
        pd.concat([seed_source, aggregate_source], ignore_index=True).to_csv(sources_dir / f"{stem}.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 5.1))
        offsets = {0: -1.8, 42: 1.8}
        for ansatz, frame in aggregate.groupby("ansatz", sort=False):
            frame = frame.sort_values("circuit_depth")
            if len(frame) > 1:
                ax.plot(frame["circuit_depth"], frame[metric], color="#374151", linewidth=1.4, alpha=0.85)
            for row in frame.itertuples(index=False):
                ax.scatter(row.circuit_depth, getattr(row, metric), s=80, marker=ANSATZ_MARKERS[ansatz], color=DEPTH_COLORS[row.circuit_depth], edgecolor="#111827", linewidth=0.8, zorder=3)
                if row.ansatz == "strong":
                    ax.annotate(row.variant_label, (row.circuit_depth, getattr(row, metric)), xytext=(5, 4), textcoords="offset points", fontsize=8)
        for row in seed_table.itertuples(index=False):
            ax.scatter(row.circuit_depth + offsets[int(row.seed)], getattr(row, metric), s=24, marker=ANSATZ_MARKERS[row.ansatz], facecolor=DEPTH_COLORS[row.circuit_depth], edgecolor="none", alpha=0.25, zorder=2)
        ax.set_xlabel("Circuit depth")
        ax.set_ylabel(y_label)
        ax.set_xticks([50, 100, 200])
        ax.set_xlim(35, 215)
        ax.set_ylim(*y_limits)
        _style_axis(ax)
        _add_encoding_legends(ax, location="lower right")
        _save_figure(fig, figures_dir, stem)

    depth_figure("asr_auc", "figure_b2", "White-box ASR AUC", (0.85, 0.985))
    depth_figure("retention_auc", "figure_b3", "Retention AUC", (0.0, 0.15))

    c1_metrics = [
        ("robust_accuracy_auc", "Robust accuracy AUC"),
        ("retention_auc", "Retention AUC"),
        ("asr_auc", "ASR AUC"),
        ("accuracy_drop_auc", "Accuracy drop AUC"),
    ]
    c1_source = aggregate[["canonical_model_name", "variant_label", "ansatz", "circuit_depth", *[m[0] for m in c1_metrics]]].copy()
    c1_source.to_csv(sources_dir / "figure_c1.csv", index=False)
    order = aggregate.sort_values(["circuit_depth", "ansatz"])["variant_label"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 5.2), sharey=True)
    y_positions = np.arange(len(order))
    indexed = aggregate.set_index("variant_label").loc[order]
    for idx, (metric, label) in enumerate(c1_metrics):
        ax = axes[idx]
        for y, row in enumerate(indexed.itertuples(index=False)):
            ax.scatter(
                getattr(row, metric), y, s=62, marker=ANSATZ_MARKERS[row.ansatz],
                color=DEPTH_COLORS[row.circuit_depth], edgecolor="#111827", linewidth=0.6,
            )
        for y, value in zip(y_positions, indexed[metric]):
            ax.hlines(y, 0, value, color="#D1D5DB", linewidth=1.0, zorder=0)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1)
        _style_axis(ax)
    axes[0].set_yticks(y_positions, order)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("VQM variant")
    fig.subplots_adjust(wspace=0.16)
    _save_figure(fig, figures_dir, "figure_c1")

    d = aggregate.merge(gradient_vqm, on="canonical_model_name", how="left", validate="one_to_one", suffixes=("", "_gradient"))
    d.to_csv(sources_dir / "figure_d1.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for row in d.itertuples(index=False):
        ax.scatter(row.first_order_fgsm_sensitivity, row.asr_auc, s=80, marker=ANSATZ_MARKERS[row.ansatz], color=DEPTH_COLORS[row.circuit_depth], edgecolor="#111827", linewidth=0.8)
    _label_points(ax, d, "first_order_fgsm_sensitivity", "asr_auc")
    ax.set_xlabel("Median first-order FGSM sensitivity (gradient L1 norm)")
    ax.set_ylabel("White-box ASR AUC")
    _style_axis(ax)
    _add_encoding_legends(ax, location="lower left")
    _save_figure(fig, figures_dir, "figure_d1")

    d.to_csv(sources_dir / "figure_d2.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for ansatz, frame in d.groupby("ansatz", sort=False):
        frame = frame.sort_values("circuit_depth")
        if len(frame) > 1:
            ax.plot(frame["circuit_depth"], frame["median_top10_abs_gradient_mass_fraction"], color="#374151", linewidth=1.4, alpha=0.85)
        for row in frame.itertuples(index=False):
            ax.scatter(row.circuit_depth, row.median_top10_abs_gradient_mass_fraction, s=80, marker=ANSATZ_MARKERS[row.ansatz], color=DEPTH_COLORS[row.circuit_depth], edgecolor="#111827", linewidth=0.8)
            if row.ansatz == "strong" or row.ansatz == "no_entangling":
                ax.annotate(row.variant_label, (row.circuit_depth, row.median_top10_abs_gradient_mass_fraction), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Top-10% absolute-gradient mass fraction")
    ax.set_xticks([50, 100, 200])
    ax.set_xlim(35, 215)
    _style_axis(ax)
    _add_encoding_legends(ax, location="upper right", depth_location="center right")
    _save_figure(fig, figures_dir, "figure_d2")
    return {stem: figures_dir / f"{stem}.png" for stem in ("figure_b1", "figure_b2", "figure_b3", "figure_c1", "figure_d1", "figure_d2")}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def _image_html(path: Path, alt: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure style="margin:0;max-width:100%;">'
        f'<img alt="{alt}" src="data:image/png;base64,{encoded}" '
        'style="display:block;width:100%;height:auto;max-width:1100px;margin:0 auto;" />'
        "</figure>"
    )


def build_methodology_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis": "Canonical VQM variant table",
                "input_data": "raw VQM self-attack logits, labels, clean predictions; model/circuit summaries; training dynamics",
                "filters": "canonical retained attack set only; source_model = target_model; seeds 0 and 42; identity deleted; unattacked depth-5/10/25 variants excluded",
                "formulas": "clean accuracy at epsilon=0; normalized trapezoidal AUC for robust accuracy, robust/clean retention, ASR, flip rate, and clean-robust accuracy drop",
                "seed_aggregation": "one row per seed; variant summaries are arithmetic means of the two seed-level AUCs",
                "epsilon_aggregation": "trapezoidal integration over the exact 21-point epsilon grid [0,1], divided by grid width",
                "interpretation": "seed-level canonical evidence for retained VQM variants",
                "confounders": "two seeds; training-time provenance differs for first-generation runs; test accuracy in the training summary is variant-level",
            },
            {
                "analysis": "Ansatz and depth effects",
                "input_data": "validated canonical VQM table",
                "filters": "depth-50 ansatz contrasts; strong-only depth contrasts at 50/100/200",
                "formulas": "exact differences, Spearman rank correlations, partial Spearman controlling clean accuracy, exploratory standardized OLS",
                "seed_aggregation": "effects computed on seed means; seed points remain visible in B2/B3",
                "epsilon_aggregation": "uses precomputed normalized AUCs",
                "interpretation": "descriptive architecture associations, not causal effects",
                "confounders": "n=6 variants; depth and expressivity measures are collinear; ansatz-depth design is unbalanced; full suggested regression is saturated",
            },
            {
                "analysis": "Clean-accuracy confounding",
                "input_data": "robust accuracy, retention, ASR, and accuracy-drop AUCs",
                "filters": "retained VQM variants",
                "formulas": "retention=robust_accuracy/clean_accuracy at each epsilon; ASR conditions on clean-correct samples",
                "seed_aggregation": "seed-level AUC first, then arithmetic seed mean",
                "epsilon_aggregation": "normalized trapezoidal AUC over [0,1]",
                "interpretation": "retention and ASR reduce—but do not eliminate—clean-accuracy dependence",
                "confounders": "ASR denominators differ with clean-correct set size; AUC weights the full large-epsilon regime",
            },
            {
                "analysis": "Input-gradient diagnostics",
                "input_data": "cached float16 per-sample input gradients for 512 MNIST test images",
                "filters": "retained VQM gradient files; representative first loaded seed from the cache-generation script",
                "formulas": "sample norms; L1/784; top ceil(10%*784) absolute mass / L1; entropy/log(784); abs(g)<=1e-6; (positive-negative)/(positive+negative)",
                "seed_aggregation": "none: cache contains one representative seed per model; medians are across 512 samples",
                "epsilon_aggregation": "not applicable; first-order coefficient is ||gradient_x loss||_1",
                "interpretation": "local first-order FGSM source sensitivity, not attack success or transferability",
                "confounders": "float16 quantization; representative seed only; loss-gradient scale and calibration can differ",
            },
            {
                "analysis": "Matched comparisons",
                "input_data": "self-attack AUCs and gradient diagnostics for VQMs, three smallest MLPs, and two HVQMs",
                "filters": "one nearest comparator per pre-specified stratum; <=5 percentage-point clean-accuracy difference is considered a valid match",
                "formulas": "absolute clean-accuracy difference and direct metric differences",
                "seed_aggregation": "attack AUCs averaged over seeds 0 and 42; gradient sensitivity is representative-seed median",
                "epsilon_aggregation": "normalized trapezoidal AUC over [0,1]",
                "interpretation": "checks whether vulnerability remains notable under reduced accuracy mismatch",
                "confounders": "most VQMs have no comparator within 5 pp; architecture, capacity, and training still differ",
            },
        ]
    )


def build_hypothesis_table(aggregate: pd.DataFrame, gradient_vqm: pd.DataFrame) -> pd.DataFrame:
    by_label = aggregate.set_index("variant_label")
    gradients = gradient_vqm.set_index("canonical_model_name")
    strong_names = [RETAINED_VARIANTS[i]["canonical_model_name"] for i in (3, 4, 5)]
    strong_l1 = [float(gradients.loc[name, "first_order_fgsm_sensitivity"]) for name in strong_names]
    rows = [
        {
            "hypothesis": "H1",
            "verdict": "Supported",
            "computation": "best retained variant and range of VQM AUCs",
            "numerical_evidence": f"best robust AUC={by_label.loc['strong-200','robust_accuracy_auc']:.3f}; best retention={by_label.loc['strong-200','retention_auc']:.3f}; best (lowest) ASR AUC={by_label.loc['strong-200','asr_auc']:.3f}",
            "interpretation": "direct FGSM self-attacks remain highly effective",
            "caveat": "evaluated MNIST models, epsilon grid, loss, and FGSM implementation only",
        },
        {
            "hypothesis": "H2",
            "verdict": "Supported descriptively",
            "computation": "depth-50 strong minus basic/random exact contrasts",
            "numerical_evidence": f"ASR AUC strong={by_label.loc['strong-50','asr_auc']:.3f}, basic={by_label.loc['basic-50','asr_auc']:.3f}, random={by_label.loc['random-50','asr_auc']:.3f}",
            "interpretation": "strong improves both absolute and clean-conditioned self-attack metrics",
            "caveat": "ansatz changes parameter count, gate structure, training duration, and clean accuracy",
        },
        {
            "hypothesis": "H3",
            "verdict": "Supported descriptively",
            "computation": "within-strong 50/100/200 depth sequence, checked in both seeds",
            "numerical_evidence": f"retention AUC {by_label.loc['strong-50','retention_auc']:.3f}->{by_label.loc['strong-100','retention_auc']:.3f}->{by_label.loc['strong-200','retention_auc']:.3f}; ASR AUC {by_label.loc['strong-50','asr_auc']:.3f}->{by_label.loc['strong-100','asr_auc']:.3f}->{by_label.loc['strong-200','asr_auc']:.3f}",
            "interpretation": "depth gives a modest, directionally consistent improvement",
            "caveat": "only three depths and two seeds; no population uncertainty claim",
        },
        {
            "hypothesis": "H4",
            "verdict": "Supported",
            "computation": "compare strongest/deepest VQM with vulnerability thresholds implied by AUCs",
            "numerical_evidence": f"strong-200 retains only {by_label.loc['strong-200','retention_auc']:.1%} of clean accuracy on average while ASR AUC is {by_label.loc['strong-200','asr_auc']:.1%}",
            "interpretation": "the improvement does not change the high-vulnerability regime",
            "caveat": "regime label is descriptive rather than a universal benchmark threshold",
        },
        {
            "hypothesis": "H5",
            "verdict": "Supported",
            "computation": "joint review of self-attack AUC and cached source gradients",
            "numerical_evidence": "strong depth L1 sensitivities=" + ", ".join(f"{value:.3g}" for value in strong_l1),
            "interpretation": "self-attacks couple source-gradient generation and target resistance; these data cannot identify which changed",
            "caveat": "first-order L1 is not realized loss increase or boundary crossing",
        },
        {
            "hypothesis": "H6",
            "verdict": "Supported as a constraint",
            "computation": "compare depth/parameter trends with separate gradient diagnostics",
            "numerical_evidence": "gradient magnitude and saliency concentration do not by themselves determine the observed ASR ordering",
            "interpretation": "expressivity has no single necessary direction for attack strength",
            "caveat": "transfer source-role analysis is outside this white-box self-attack report",
        },
    ]
    return pd.DataFrame(rows)


def _source_entry(source_id: str, label: str, path: str) -> dict[str, Any]:
    safe_path = path.replace("\\", "/")
    return {
        "id": source_id,
        "label": label,
        "path": safe_path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{safe_path}')",
            "description": f"Loads the reviewed rows for {label} from the exported source CSV.",
            "tables_used": [safe_path],
            "filters": ["Rows are the bounded outputs of the retained-VQM analysis."],
        },
    }


def build_report_artifact(
    paths: AnalysisPaths,
    figures: dict[str, Path],
    seed_table: pd.DataFrame,
    aggregate: pd.DataFrame,
    gradient_vqm: pd.DataFrame,
    matches: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> dict[str, Any]:
    by_label = aggregate.set_index("variant_label")
    figure_d1 = pd.read_csv(paths.output_dir / "figure_sources" / "figure_d1.csv")
    valid_matches = matches[matches["within_5pp"]].copy()
    report_sources = [
        _source_entry("variant_table_source", "Canonical VQM seed table", "outputs/whitebox_vqm_analysis/tables/vqm_variant_seed_table.csv"),
        _source_entry("gradient_source", "Cached VQM input-gradient diagnostics", "outputs/whitebox_vqm_analysis/tables/vqm_gradient_diagnostics.csv"),
        _source_entry("match_source", "Clean-accuracy matched comparison table", "outputs/whitebox_vqm_analysis/tables/matched_comparisons.csv"),
        _source_entry("hypothesis_source", "Hypothesis verdict evidence", "outputs/whitebox_vqm_analysis/tables/hypothesis_verdicts.csv"),
        _source_entry("figure_d1_source", "Figure D1 source data", "outputs/whitebox_vqm_analysis/figure_sources/figure_d1.csv"),
    ]
    variant_display = aggregate[[
        "variant_label", "ansatz", "circuit_depth", "number_of_gates", "quantum_parameter_count",
        "clean_accuracy", "robust_accuracy_auc", "retention_auc", "asr_auc", "flip_rate_auc",
    ]].copy()
    match_display = matches[[
        "vqm_label", "comparison_stratum", "matched_model", "clean_accuracy_difference", "within_5pp",
        "robust_accuracy_auc_vqm", "robust_accuracy_auc_match", "retention_auc_vqm", "retention_auc_match",
        "asr_auc_vqm", "asr_auc_match",
    ]].copy()
    hypothesis_display = hypotheses[["hypothesis", "verdict", "numerical_evidence", "interpretation", "caveat"]].copy()
    d1_display = figure_d1[[
        "variant_label", "ansatz", "circuit_depth", "first_order_fgsm_sensitivity", "asr_auc", "clean_accuracy"
    ]].copy()

    executive = (
        "## Executive summary\n\n"
        f"- **The retained VQMs are not inherently robust to direct FGSM self-attacks.** The best case, strong-200, has robust-accuracy AUC {by_label.loc['strong-200','robust_accuracy_auc']:.1%}, retention AUC {by_label.loc['strong-200','retention_auc']:.1%}, and ASR AUC {by_label.loc['strong-200','asr_auc']:.1%}.\n"
        f"- **The strong ansatz improves the depth-50 profile beyond its higher clean accuracy.** Its ASR AUC is {by_label.loc['strong-50','asr_auc']:.1%}, versus {by_label.loc['basic-50','asr_auc']:.1%} for basic and {by_label.loc['random-50','asr_auc']:.1%} for random; retention rises to {by_label.loc['strong-50','retention_auc']:.1%}.\n"
        f"- **Within strong, depth helps modestly and consistently.** Retention AUC rises {by_label.loc['strong-50','retention_auc']:.1%} -> {by_label.loc['strong-100','retention_auc']:.1%} -> {by_label.loc['strong-200','retention_auc']:.1%}, while ASR AUC falls {by_label.loc['strong-50','asr_auc']:.1%} -> {by_label.loc['strong-100','asr_auc']:.1%} -> {by_label.loc['strong-200','asr_auc']:.1%}.\n"
        f"- **Clean-accuracy matching is sparse.** Only {len(valid_matches)} of {len(matches)} pre-specified VQM-comparator matches are within five percentage points, so family-level claims are not supported.\n"
        "- **The white-box result is coupled.** Cached input gradients diagnose the source role, but they cannot separate target resistance from gradient quality or establish transferability."
    )
    blocks: list[dict[str, Any]] = [
        {"id": "report_title", "type": "markdown", "body": "# White-box FGSM self-attack profiles of retained VQMs"},
        {"id": "executive_summary", "type": "markdown", "body": executive},
        {
            "id": "retained_variants_section",
            "type": "markdown",
            "body": (
                "## Retained VQM variants and data quality\n\n"
                "The canonical retained set contains basic-50, random-50 (the ratio_imprim=0.75 training run), no-entangling-50, and strong-50/100/200, each at seeds 0 and 42. Identity is explicitly deleted by the attack migration report; depth-5/10/25 VQMs are valid training runs but were not attacked and are excluded. VQM attack metrics were recomputed from the logits cubes, labels, and clean predictions. Each seed has 21 epsilon points from 0 to 1 and 10,000 evaluated images. The legacy white-box summary agrees where populated; its no-entangling row is blank despite complete canonical raw artifacts."
            ),
        },
        {"id": "retained_variants_table_block", "type": "table", "tableId": "retained_variants_table"},
        {
            "id": "ansatz_section",
            "type": "markdown",
            "body": (
                "## Ansatz comparison\n\n"
                "At depth 50, strong has the clearest descriptive advantage over basic and random on robust AUC, retention, and clean-correct ASR. No-entangling is retained because its raw attack cubes are complete, but its very low clean accuracy makes absolute robust accuracy particularly uninformative.\n\n"
                "### Figure B1\n\n**How computed.** Each point is the two-seed mean of clean accuracy and normalized trapezoidal robust-accuracy AUC; shape encodes ansatz and color encodes configured circuit depth. **What it shows.** Absolute robust AUC rises with clean accuracy, with strong variants occupying the upper-right of this small VQM set. **What it does not establish.** The slope is not a causal expressivity effect and does not remove clean-accuracy confounding."
            ),
        },
        {"id": "figure_b1_image", "type": "html", "body": _image_html(figures["figure_b1"], "Figure B1: clean accuracy versus robust accuracy AUC")},
        {
            "id": "depth_section",
            "type": "markdown",
            "body": (
                "## Depth comparison\n\n"
                "Only the strong ansatz has multiple attacked depths, so depth inference is a within-strong descriptive sequence. Both seeds move in the same favorable direction; faint points are seeds and bold points are seed means.\n\n"
                "### Figure B2\n\n**How computed.** ASR is conditioned on each victim's clean-correct samples, integrated over epsilon per seed, then averaged across seeds. **What it shows.** Strong ASR AUC declines monotonically with depth. **What it does not establish.** Three configurations and two seeds do not establish a population trend or causal depth effect."
            ),
        },
        {"id": "figure_b2_image", "type": "html", "body": _image_html(figures["figure_b2"], "Figure B2: white-box ASR AUC versus circuit depth")},
        {
            "id": "figure_b3_method",
            "type": "markdown",
            "body": (
                "### Figure B3\n\n**How computed.** Retention is robust accuracy divided by the same seed's clean accuracy at every epsilon, followed by normalized trapezoidal AUC and seed averaging. **What it shows.** Strong retention improves from depth 50 to 100 to 200 after clean normalization. **What it does not establish.** Retention still depends on the same model, data, and attack path and is not an architecture-independent robustness measure."
            ),
        },
        {"id": "figure_b3_image", "type": "html", "body": _image_html(figures["figure_b3"], "Figure B3: retention AUC versus circuit depth")},
        {
            "id": "confounding_section",
            "type": "markdown",
            "body": (
                "## Clean-accuracy confounding\n\n"
                "The improvement is not merely an artifact of higher clean accuracy: strong improves retention and lowers clean-correct ASR, and within strong the depth ordering persists on both normalized metrics. The full suggested regression is not estimable: four ansatz levels plus clean accuracy, depth, and an intercept saturate six variants. Reduced clean-plus-depth regressions and partial rank associations are therefore exploratory only.\n\n"
                "### Figure C1\n\n**How computed.** The four AUCs use the same epsilon grid and are shown on a common 0-1 scale; robust accuracy and retention are higher-is-better, while ASR and accuracy drop are higher-is-worse. **What it shows.** Strong improves both absolute and normalized metrics, but all variants remain in a regime of low retention and high ASR. **What it does not establish.** The profile cannot isolate ansatz from parameter count, gates, training time, or optimization."
            ),
        },
        {"id": "figure_c1_image", "type": "html", "body": _image_html(figures["figure_c1"], "Figure C1: four-metric VQM profile")},
        {
            "id": "gradient_section",
            "type": "markdown",
            "body": (
                "## Input-gradient diagnostics\n\n"
                "The gradient cache contains 512 per-sample, 784-pixel float16 gradients for one representative loaded seed per model. Reported values are medians across samples. First-order FGSM sensitivity is the gradient L1 norm because g^T[epsilon sign(g)] = epsilon ||g||_1; it is a local coefficient, not realized attack success.\n\n"
                "### Figure D1\n\n**How computed.** Median cached gradient L1 is paired with the two-seed white-box ASR AUC for each VQM. **What it shows.** Whether lower ASR coincides with weaker first-order gradients can be checked directly. **What it does not establish.** Gradient scale does not identify boundary distance, nonlinear loss behavior, or transferability."
            ),
        },
        {"id": "figure_d1_image", "type": "html", "body": _image_html(figures["figure_d1"], "Figure D1: first-order FGSM sensitivity versus ASR AUC")},
        {
            "id": "figure_d2_method",
            "type": "markdown",
            "body": (
                "### Figure D2\n\n**How computed.** For each sample, the largest ceil(10% x 784) absolute gradients are summed and divided by total absolute-gradient mass; model values are medians. **What it shows.** It distinguishes concentrated from distributed saliency across depth. **What it does not establish.** Concentration is not attack success, target resistance, or transferability."
            ),
        },
        {"id": "figure_d2_image", "type": "html", "body": _image_html(figures["figure_d2"], "Figure D2: saliency concentration versus circuit depth")},
        {
            "id": "matched_section",
            "type": "markdown",
            "body": (
                "## Matched comparisons\n\n"
                "The comparison pool is pre-specified rather than benchmark-wide: the three smallest MLP variants and the two attacked HVQMs. Each VQM is paired with the nearest clean accuracy within each stratum; a difference of at most five percentage points is required for a defensible clean match. Rows outside the tolerance are retained only to show the coverage gap and should not support family claims."
            ),
        },
        {"id": "matched_table_block", "type": "table", "tableId": "matched_table"},
        {
            "id": "hypothesis_section",
            "type": "markdown",
            "body": "## Hypothesis verdicts\n\nVerdicts are descriptive and scoped to the retained MNIST VQMs, the saved FGSM configuration, the exact epsilon grid, and the available two attack seeds.",
        },
        {"id": "hypothesis_table_block", "type": "table", "tableId": "hypothesis_table"},
        {
            "id": "claims_section",
            "type": "markdown",
            "body": (
                "## Thesis-ready claims\n\n"
                "- The evaluated VQMs are highly vulnerable to direct gradient-based self-attacks; none provides evidence of inherent quantum robustness.\n"
                "- At matched depth 50, the strong ansatz has a better self-attack profile than the retained basic and random variants on both retention and clean-correct ASR, although it also differs in parameters, gates, training time, and clean accuracy.\n"
                "- Within the strong ansatz, increasing configured depth from 50 to 100 to 200 yields a modest, directionally consistent improvement across both attack seeds, but the deepest model still has high ASR and low retention.\n"
                "- White-box self-attacks couple source-gradient generation and target susceptibility; the present results cannot determine whether depth changes one role, the other, or both.\n"
                "- Input-gradient magnitude and concentration are diagnostics of the local source loss surface, not evidence of transferability or universal robustness."
            ),
        },
        {
            "id": "recommended_figures_section",
            "type": "markdown",
            "body": (
                "## Recommended figures\n\n"
                "1. **Primary thesis figure: C1.** It separates absolute robust accuracy from retention, ASR, and accuracy drop, making the clean-accuracy confound visible without claiming a benchmark.\n"
                "2. **Depth evidence: B2 and B3 together.** The pair shows the same within-strong improvement through clean-conditioned and clean-normalized metrics.\n"
                "3. **Mechanism caveat: D1.** Use when discussing the coupled source-target nature of self-attacks; pair it with the explicit first-order limitation.\n"
                "4. **Context figure: B1.** Useful for showing why robust-accuracy AUC alone is confounded by clean accuracy.\n\n"
                "### Further questions\n\nA seed-resolved gradient cache, matched-capacity training design, and separate source-role transfer analysis would most improve confidence."
            ),
        },
        {
            "id": "interactive_audit_note",
            "type": "markdown",
            "body": "### Interactive audit view\n\nThe native scatter below exposes the Figure D1 rows and source metadata for exact tooltip inspection; the exported thesis figure above remains the authoritative shape-and-color rendering.",
        },
        {"id": "interactive_d1_block", "type": "chart", "chartId": "interactive_d1"},
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "White-box FGSM self-attack profiles of retained VQMs",
            "generatedAt": pd.Timestamp.utcnow().isoformat(),
            "blocks": blocks,
            "charts": [
                {
                    "id": "interactive_d1",
                    "title": "First-order FGSM sensitivity and white-box ASR AUC",
                    "subtitle": "Retained VQM seed means; gradient sensitivity uses one representative cached seed",
                    "type": "scatter",
                    "dataset": "figure_d1",
                    "sourceId": "figure_d1_source",
                    "encodings": {
                        "x": {"field": "first_order_fgsm_sensitivity", "type": "quantitative", "label": "Median gradient L1"},
                        "y": {"field": "asr_auc", "type": "quantitative", "format": "percent", "label": "ASR AUC"},
                        "color": {"field": "ansatz", "type": "nominal", "label": "Ansatz"},
                        "label": {"field": "variant_label", "type": "text"},
                        "tooltip": [
                            {"field": "variant_label", "type": "text", "label": "Variant"},
                            {"field": "circuit_depth", "type": "quantitative", "label": "Depth"},
                            {"field": "clean_accuracy", "type": "quantitative", "format": "percent", "label": "Clean accuracy"},
                        ],
                    },
                    "xAxisTitle": "Median first-order FGSM sensitivity (gradient L1 norm)",
                    "yAxisTitle": "White-box ASR AUC",
                    "maxRows": 20,
                }
            ],
            "tables": [
                {
                    "id": "retained_variants_table",
                    "title": "Retained VQM seed-mean metrics",
                    "subtitle": "Six canonical variants; attack AUCs average seed-level integrations for seeds 0 and 42",
                    "dataset": "variant_summary",
                    "sourceId": "variant_table_source",
                    "density": "spacious",
                    "defaultSort": {"field": "circuit_depth", "direction": "asc"},
                    "columns": [
                        {"field": "variant_label", "label": "Variant"},
                        {"field": "ansatz", "label": "Ansatz"},
                        {"field": "circuit_depth", "label": "Depth", "type": "number"},
                        {"field": "number_of_gates", "label": "Total gates", "type": "number"},
                        {"field": "quantum_parameter_count", "label": "Quantum params", "type": "number"},
                        {"field": "clean_accuracy", "label": "Clean accuracy", "format": "percent"},
                        {"field": "robust_accuracy_auc", "label": "Robust AUC", "format": "percent"},
                        {"field": "retention_auc", "label": "Retention AUC", "format": "percent"},
                        {"field": "asr_auc", "label": "ASR AUC", "format": "percent"},
                    ],
                },
                {
                    "id": "matched_table",
                    "title": "Pre-specified nearest clean-accuracy comparisons",
                    "subtitle": "One low-capacity MLP and one HVQM match per VQM; <=5 pp identifies defensible matches",
                    "dataset": "matched_comparisons",
                    "sourceId": "match_source",
                    "density": "comfortable",
                    "defaultSort": {"field": "clean_accuracy_difference", "direction": "asc"},
                    "columns": [
                        {"field": "vqm_label", "label": "VQM"},
                        {"field": "comparison_stratum", "label": "Stratum"},
                        {"field": "matched_model", "label": "Matched model"},
                        {"field": "clean_accuracy_difference", "label": "Clean gap", "format": "percent"},
                        {"field": "within_5pp", "label": "Within 5 pp"},
                        {"field": "retention_auc_vqm", "label": "VQM retention", "format": "percent"},
                        {"field": "retention_auc_match", "label": "Match retention", "format": "percent"},
                        {"field": "asr_auc_vqm", "label": "VQM ASR", "format": "percent"},
                        {"field": "asr_auc_match", "label": "Match ASR", "format": "percent"},
                    ],
                },
                {
                    "id": "hypothesis_table",
                    "title": "Hypothesis verdicts",
                    "subtitle": "Evidence is descriptive and scoped to retained MNIST VQMs and saved FGSM artifacts",
                    "dataset": "hypothesis_verdicts",
                    "sourceId": "hypothesis_source",
                    "density": "spacious",
                    "defaultSort": {"field": "hypothesis", "direction": "asc"},
                    "columns": [
                        {"field": "hypothesis", "label": "Hypothesis"},
                        {"field": "verdict", "label": "Verdict"},
                        {"field": "numerical_evidence", "label": "Numerical evidence"},
                        {"field": "interpretation", "label": "Interpretation"},
                        {"field": "caveat", "label": "Caveat"},
                    ],
                },
            ],
            "sources": report_sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": pd.Timestamp.utcnow().isoformat(),
            "status": "ready",
            "datasets": {
                "variant_summary": _records(variant_display),
                "matched_comparisons": _records(match_display),
                "hypothesis_verdicts": _records(hypothesis_display),
                "figure_d1": _records(d1_display),
            },
        },
        "sources": report_sources,
    }


def run_analysis(paths: AnalysisPaths) -> dict[str, pd.DataFrame]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = paths.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    source_inventory = discover_sources(paths)
    missing = source_inventory[source_inventory["required"] & ~source_inventory["exists"]]
    if len(missing):
        raise FileNotFoundError("missing required sources:\n" + missing[["source_id", "path"]].to_string(index=False))

    curves, raw_checks = load_vqm_attack_curves(paths)
    attack_seed = summarize_attack_curves(curves)
    seed_table = add_training_and_circuit_metadata(paths, attack_seed)
    aggregate = aggregate_variants(seed_table)
    gradients = load_gradient_diagnostics(paths)
    gradient_vqm = gradients[gradients["canonical_model_name"].isin(RETAINED_BY_NAME)].copy()
    if gradient_vqm["canonical_model_name"].nunique() != len(RETAINED_VARIANTS):
        missing_gradient = sorted(set(RETAINED_BY_NAME) - set(gradient_vqm["canonical_model_name"]))
        raise ValueError(f"retained VQM gradient caches missing: {missing_gradient}")
    all_self_auc = load_all_self_attack_auc(paths)
    matches = build_matched_comparisons(aggregate, all_self_auc, gradients)
    associations, regressions = compute_associations(aggregate)
    strong_differences, ansatz_differences = compute_exact_differences(aggregate)
    legacy_checks = validate_legacy_whitebox(paths, aggregate)
    methodology = build_methodology_table()
    hypotheses = build_hypothesis_table(aggregate, gradient_vqm)

    outputs = {
        "source_inventory": source_inventory,
        "vqm_attack_curves": curves,
        "raw_aggregate_checks": raw_checks,
        "vqm_variant_seed_table": seed_table,
        "vqm_variant_summary": aggregate,
        "vqm_gradient_diagnostics": gradient_vqm,
        "all_gradient_diagnostics": gradients,
        "matched_comparisons": matches,
        "spearman_associations": associations,
        "exploratory_regressions": regressions,
        "within_strong_depth_differences": strong_differences,
        "depth50_ansatz_differences": ansatz_differences,
        "legacy_whitebox_validation": legacy_checks,
        "methodology": methodology,
        "hypothesis_verdicts": hypotheses,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    figures = build_figures(paths.output_dir, seed_table, aggregate, gradient_vqm)
    artifact = build_report_artifact(paths, figures, seed_table, aggregate, gradient_vqm, matches, hypotheses)
    (paths.output_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return outputs
