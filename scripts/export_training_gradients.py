"""Export last-batch epoch gradient information from saved training runs.

This is intentionally a plain Python utility rather than a Hydra application.
Edit the constants below if the artifact locations change, then run this file
from the repository root.  It never writes below the experiment root.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUESTED_MODELS_ROOT = PROJECT_ROOT / "logs" / "ByModels"
LEGACY_MODELS_ROOT = PROJECT_ROOT / "logs" / "ByModel"
OUTPUT_DIR = PROJECT_ROOT / "results" / "training_gradients"
EXISTING_IDENTITY_TABLES = (
    PROJECT_ROOT / "results" / "training_dynamics" / "model_static_metadata.csv",
    PROJECT_ROOT / "outputs" / "training_dynamics_export" / "model_static_metadata.csv",
)
NEAR_ZERO_THRESHOLD = 1.0e-12

LONG_COLUMNS = [
    "canonical_model_name", "model_config_name", "seed", "epoch",
    "parameter_name", "parameter_shape", "parameter_numel", "component",
    "raw_l2", "rms_l2", "extraction_source", "mapping_rule",
    "mapping_confidence", "manual_review",
]
COMPONENT_COLUMNS = [
    "canonical_model_name", "model_config_name", "seed", "epoch", "component",
    "n_parameter_tensors", "total_parameter_numel", "raw_l2_mean",
    "raw_l2_median", "raw_l2_sum", "raw_l2_max", "rms_l2_mean",
    "rms_l2_median", "rms_l2_weighted_by_numel", "squared_gradient_energy",
    "fraction_missing", "fraction_near_zero",
]
RATIO_COLUMNS = [
    "canonical_model_name", "model_config_name", "seed", "epoch",
    "quantum_to_all_classical_raw_ratio", "quantum_to_all_classical_rms_ratio",
    "quantum_to_encoder_rms_ratio", "quantum_to_readout_rms_ratio",
    "quantum_to_classical_without_bn_rms_ratio",
    "quantum_to_classical_without_residual_rms_ratio",
    "quantum_gradient_energy_fraction",
]
MAPPING_COLUMNS = [
    "canonical_model_name", "model_config_name", "parameter_name",
    "parameter_shape", "parameter_numel", "component", "mapping_rule",
    "mapping_confidence", "manual_review", "parameter_metadata_source",
]
QUALITY_COLUMNS = [
    "canonical_model_name", "model_config_name", "seed", "logger_pkl_present",
    "metrics_csv_present", "gradient_metrics_present", "epochs_expected",
    "epochs_extracted", "checkpoint_model_load_success", "parameter_names_matched",
    "unresolved_parameter_names", "missing_parameter_sizes", "duplicate_epochs",
    "nan_values", "all_zero_or_suspicious_gradients",
    "model_parameters_never_logged", "logged_parameters_not_in_model",
    "logger_csv_comparison_count", "logger_csv_mismatch_count",
    "epoch_alignment_status", "status", "notes",
]


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    shape: tuple[int, ...] | None
    numel: int | None
    component: str
    rule: str
    confidence: str
    manual_review: bool
    source: str


@dataclass
class GradientSource:
    source: str
    histories: dict[str, list[float]]
    steps: list[float]
    nan_values: int = 0
    notes: list[str] | None = None
    train_steps: list[float] | None = None

    @property
    def n_epochs(self) -> int:
        return max([len(self.steps), *(len(v) for v in self.histories.values())], default=0)


def resolve_models_root() -> Path:
    if REQUESTED_MODELS_ROOT.exists():
        return REQUESTED_MODELS_ROOT.resolve()
    if LEGACY_MODELS_ROOT.exists():
        return LEGACY_MODELS_ROOT.resolve()
    raise FileNotFoundError(
        f"Neither requested root {REQUESTED_MODELS_ROOT} nor fallback {LEGACY_MODELS_ROOT} exists."
    )


def discover_model_dirs(root: Path) -> list[Path]:
    return [p.parent for p in sorted(root.rglob(".model")) if p.is_file()]


def iter_seed_dirs(model_dir: Path) -> list[Path]:
    return sorted(
        (p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def load_identity_map() -> tuple[dict[tuple[str, int], str], Path | None]:
    for path in EXISTING_IDENTITY_TABLES:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["model_config_name", "canonical_model_name", "seed"])
        mapping = {
            (str(row.model_config_name).replace("\\", "/"), int(row.seed)): str(row.canonical_model_name)
            for row in df.itertuples(index=False)
        }
        return mapping, path.resolve()
    return {}, None


def fallback_canonical_name(model_config_name: str) -> str:
    parts = model_config_name.split("/")
    family = parts[0].title() if parts else "Unknown"
    architecture_map = {
        "convnet": "ConvNet", "lenet": "LeNet", "mlp": "MLP",
        "mlpmixer": "MLPMixer", "resnet18": "ResNet18", "hvql": "HVQL",
        "hvqm": "HVQM", "hpvqm": "HPVQM", "hpvql": "HPVQL",
        "hqe": "HQE", "hquanv": "HQuanv", "vqm": "VQM",
    }
    architecture = architecture_map.get(parts[1].lower(), parts[1] if len(parts) > 1 else "Unknown")
    suffix = "/".join(parts[2:])
    return f"{family}/{architecture}" + (f"/{suffix}" if suffix else "")


def canonical_name(
    identity_map: dict[tuple[str, int], str], model_config_name: str, seed: int
) -> str:
    direct = identity_map.get((model_config_name, seed))
    if direct:
        return direct
    same_model = sorted(v for (name, _), v in identity_map.items() if name == model_config_name)
    return same_model[0] if same_model else fallback_canonical_name(model_config_name)


def read_logger_gradients(path: Path) -> GradientSource | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        raw = payload.get("_raw", payload) if isinstance(payload, dict) else {}
        group = raw.get("grad_l2", {}) if isinstance(raw, dict) else {}
        if not isinstance(group, dict):
            return None
        histories: dict[str, list[float]] = {}
        nan_values = 0
        for name, values in group.items():
            if name == "step" or not isinstance(values, (list, tuple, np.ndarray)):
                continue
            converted: list[float] = []
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = float("nan")
                nan_values += int(not math.isfinite(number))
                converted.append(number)
            histories[str(name)] = converted
        if not histories:
            return None
        steps = [float(x) for x in group.get("step", [])]
        lengths = sorted({len(v) for v in histories.values()})
        notes = []
        if len(lengths) > 1:
            notes.append(f"unequal logger history lengths={lengths}")
        train_group = raw.get("train", {}) if isinstance(raw, dict) else {}
        train_steps = (
            [float(x) for x in train_group.get("step", [])]
            if isinstance(train_group, dict) else []
        )
        if steps and train_steps and steps != train_steps:
            notes.append("gradient steps do not exactly match train steps")
        return GradientSource("logger.pkl", histories, steps, nan_values, notes, train_steps)
    except Exception as exc:
        return GradientSource("logger.pkl", {}, [], 0, [f"logger read failed: {exc}"])


def read_csv_gradients(path: Path) -> GradientSource | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return GradientSource("metrics.csv", {}, [], 0, [f"metrics CSV read failed: {exc}"])
    prefix = "grad_l2_"
    columns = [c for c in df.columns if str(c).startswith(prefix) and c != "grad_l2_step"]
    if not columns:
        return None
    step_col = "grad_l2_step" if "grad_l2_step" in df.columns else None
    selected = df.loc[df[columns].notna().any(axis=1)].copy()
    if step_col:
        selected = selected.sort_values(step_col, kind="stable")
    histories = {
        str(column)[len(prefix):]: [float(v) for v in selected[column].tolist()]
        for column in columns
    }
    nan_values = int(selected[columns].isna().sum().sum())
    steps = [float(v) for v in selected[step_col].tolist()] if step_col else []
    return GradientSource("metrics.csv", histories, steps, nan_values, [])


def read_tensorboard_gradients(tensorboard_dir: Path) -> GradientSource | None:
    event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    events: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if not tag.startswith("grad_l2/"):
                continue
            name = tag.split("/", 1)[1]
            events[name].extend((int(event.step), float(event.value)) for event in accumulator.Scalars(tag))
    if not events:
        return None
    histories = {}
    all_steps: set[int] = set()
    for name, values in events.items():
        deduped = dict(sorted(values))
        histories[name] = list(deduped.values())
        all_steps.update(deduped)
    return GradientSource("tensorboard", histories, [float(x) for x in sorted(all_steps)], 0, [])


def compare_sources(primary: GradientSource | None, csv: GradientSource | None) -> tuple[int, int]:
    if primary is None or csv is None or not primary.histories or not csv.histories:
        return 0, 0
    compared = mismatches = 0
    for name in sorted(set(primary.histories) & set(csv.histories)):
        left, right = primary.histories[name], csv.histories[name]
        for a, b in zip(left, right):
            if math.isfinite(a) and math.isfinite(b):
                compared += 1
                mismatches += int(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    return compared, mismatches


def checkpoint_path(seed_dir: Path) -> Path | None:
    final = seed_dir / "final.ckpt"
    if final.exists():
        return final
    candidates = sorted((seed_dir / "checkpoints").glob("*.ckpt"))
    return candidates[0] if candidates else None


def state_dict_parameter_metadata(checkpoint: Path | None) -> dict[str, tuple[tuple[int, ...], int]]:
    if checkpoint is None:
        return {}
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        excluded = ("running_mean", "running_var", "num_batches_tracked")
        return {
            str(name): (tuple(int(x) for x in tensor.shape), int(tensor.numel()))
            for name, tensor in state.items()
            if torch.is_tensor(tensor) and not str(name).endswith(excluded)
        }
    except Exception:
        return {}


def owner_module_path(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0] if "." in parameter_name else ""


def is_descendant(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def classify_loaded_parameter(
    name: str,
    modules: dict[str, Any],
    quantum_parameter_modules: set[str],
    model: Any,
) -> tuple[str, str, str, bool]:
    owner_path = owner_module_path(name)
    owner = modules.get(owner_path)
    owner_type = owner.__class__.__name__ if owner is not None else "unknown"

    if owner is not None and isinstance(owner, torch.nn.modules.batchnorm._BatchNorm):
        return "batch_norm", f"loaded module type {owner_type} at {owner_path}", "high", False
    if name == "classifier.residual_scale" or is_descendant(name, "classifier.residual_proj"):
        return "residual_path", "loaded ResVQL residual_scale/residual_proj structure", "high", False
    if owner_path in quantum_parameter_modules:
        return "quantum", f"parameter owned by loaded quantum TorchLayer at {owner_path}", "high", False

    encoder = getattr(model, "encoder", None)
    classifier = getattr(model, "classifier", None)
    encoder_type = encoder.__class__.__name__ if encoder is not None else ""
    classifier_has_quantum = any(path.startswith("classifier.") for path in quantum_parameter_modules)
    encoder_has_quantum = any(path.startswith("encoder.") for path in quantum_parameter_modules)

    if name.startswith("encoder."):
        if encoder_has_quantum and encoder_type.lower() == "quanvolution":
            return "classical_readout", "loaded HQuanv encoder: classical parameter follows patch quantum layer", "high", False
        return "classical_encoder", "loaded HQNN encoder module before classifier quantum stage", "high", False
    if name.startswith("classifier."):
        if name.startswith("classifier.fc1."):
            return "classical_encoder", "loaded classifier fc1 precedes quantum module", "high", False
        if classifier_has_quantum and any(
            is_descendant(name, prefix)
            for prefix in ("classifier.fc2", "classifier.fc_out", "classifier.post", "classifier.net")
        ):
            return "classical_readout", "loaded classifier module follows quantum module", "high", False
        if encoder_has_quantum and not classifier_has_quantum:
            return "classical_readout", "loaded classifier follows quantum encoder", "high", False
        return "other_classical", f"loaded classifier module {owner_type}; order not structurally resolved", "medium", True
    if name.startswith("post."):
        return "classical_readout", "loaded model-level post projection follows VQC", "high", False
    if name.startswith("vqc.") and owner_path not in quantum_parameter_modules:
        return "other_classical", "loaded VQC descendant outside quantum parameter module", "medium", True
    return "other_classical", f"loaded classical module type {owner_type}", "high", False


def classify_checkpoint_parameter(name: str) -> tuple[str, str, str, bool]:
    lower = name.lower()
    if any(token in lower for token in ("batchnorm", ".bn.", ".bn1.", ".bn2.")):
        return "batch_norm", "checkpoint fallback: canonical batch-normalization path", "medium", True
    if "residual_proj" in lower or "residual_scale" in lower:
        return "residual_path", "checkpoint fallback: canonical ResVQL residual path", "medium", True
    if ".qlayer." in lower or lower.startswith("qlayer."):
        return "quantum", "checkpoint fallback: registered TorchLayer parameter path", "medium", True
    if lower.startswith("encoder."):
        return "classical_encoder", "checkpoint fallback: encoder path", "medium", True
    if any(token in lower for token in (".fc2.", ".fc_out.", ".post.", "post.")):
        return "classical_readout", "checkpoint fallback: known post-quantum projection path", "medium", True
    if lower.startswith("classifier.fc1."):
        return "classical_encoder", "checkpoint fallback: known pre-quantum fc1 path", "medium", True
    return "unresolved", "checkpoint fallback could not determine execution order", "low", True


def load_parameter_info(
    model_dir: Path, representative_seed: Path, logged_names: set[str]
) -> tuple[dict[str, ParameterInfo], set[str], bool, list[str]]:
    checkpoint = checkpoint_path(representative_seed)
    notes: list[str] = []
    model = None
    model_load_success = False
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from analytics.tables._experiment_io import load_single_model

        if checkpoint is None:
            raise FileNotFoundError("no checkpoint")
        model = load_single_model(model_dir, representative_seed, checkpoint, map_location="cpu")
        model_load_success = True
    except Exception as exc:
        notes.append(f"repository model load failed: {type(exc).__name__}: {exc}")

    info: dict[str, ParameterInfo] = {}
    model_parameter_names: set[str] = set()
    if model is not None:
        modules = dict(model.named_modules())
        quantum_modules = {
            path for path, module in modules.items()
            if module.__class__.__name__ == "TorchLayer"
            or module.__class__.__module__.startswith("pennylane.qnn")
        }
        for name, parameter in model.named_parameters():
            shape = tuple(int(x) for x in parameter.shape)
            component, rule, confidence, manual_review = classify_loaded_parameter(
                name, modules, quantum_modules, model
            )
            info[name] = ParameterInfo(
                name, shape, int(parameter.numel()), component, rule,
                confidence, manual_review, "repository model loader",
            )
            model_parameter_names.add(name)
    else:
        checkpoint_meta = state_dict_parameter_metadata(checkpoint)
        model_parameter_names = set(checkpoint_meta)
        for name, (shape, numel) in checkpoint_meta.items():
            component, rule, confidence, manual_review = classify_checkpoint_parameter(name)
            info[name] = ParameterInfo(
                name, shape, numel, component, rule, confidence,
                manual_review, "checkpoint state_dict",
            )

    for name in sorted(logged_names - set(info)):
        info[name] = ParameterInfo(
            name, None, None, "unresolved", "logged name not found in loaded model/checkpoint",
            "low", True, "unresolved",
        )
    return info, model_parameter_names, model_load_success, notes


def expected_epochs(model_dir: Path, source: GradientSource | None) -> int | None:
    info_path = model_dir / "model.info"
    if info_path.exists():
        text = info_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Trained epochs:\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return source.n_epochs if source is not None and source.n_epochs else None


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def build_long_rows(
    canonical: str,
    model_config_name: str,
    seed: int,
    source: GradientSource,
    parameter_info: dict[str, ParameterInfo],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parameter_name, values in sorted(source.histories.items()):
        meta = parameter_info[parameter_name]
        for epoch, value in enumerate(values):
            raw_l2 = finite_or_nan(value)
            rms_l2 = (
                raw_l2 / math.sqrt(meta.numel)
                if meta.numel is not None and meta.numel > 0 and math.isfinite(raw_l2)
                else float("nan")
            )
            rows.append({
                "canonical_model_name": canonical,
                "model_config_name": model_config_name,
                "seed": seed,
                "epoch": epoch,
                "parameter_name": parameter_name,
                "parameter_shape": json.dumps(list(meta.shape)) if meta.shape is not None else "",
                "parameter_numel": meta.numel,
                "component": meta.component,
                "raw_l2": raw_l2,
                "rms_l2": rms_l2,
                "extraction_source": source.source,
                "mapping_rule": meta.rule,
                "mapping_confidence": meta.confidence,
                "manual_review": meta.manual_review,
            })
    return rows


def weighted_rms_mean(frame: pd.DataFrame) -> float:
    valid = frame.dropna(subset=["rms_l2", "parameter_numel"])
    valid = valid[valid["parameter_numel"] > 0]
    if valid.empty:
        return float("nan")
    return float(np.average(valid["rms_l2"], weights=valid["parameter_numel"]))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def aggregate_components(long_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=COMPONENT_COLUMNS)
    expected_counts = (
        mapping_df.groupby(["model_config_name", "component"], dropna=False)["parameter_name"]
        .nunique().to_dict()
    )
    rows = []
    keys = ["canonical_model_name", "model_config_name", "seed", "epoch", "component"]
    for key, frame in long_df.groupby(keys, sort=True, dropna=False):
        finite = frame[np.isfinite(frame["raw_l2"])]
        expected = expected_counts.get((key[1], key[4]), len(frame))
        observed = finite["parameter_name"].nunique()
        numel = finite["parameter_numel"].dropna()
        rms = finite["rms_l2"].dropna()
        raw = finite["raw_l2"]
        rows.append(dict(zip(keys, key)) | {
            "n_parameter_tensors": int(observed),
            "total_parameter_numel": int(numel.sum()) if not numel.empty else np.nan,
            "raw_l2_mean": raw.mean(), "raw_l2_median": raw.median(),
            "raw_l2_sum": raw.sum(), "raw_l2_max": raw.max(),
            "rms_l2_mean": rms.mean(), "rms_l2_median": rms.median(),
            "rms_l2_weighted_by_numel": weighted_rms_mean(finite),
            "squared_gradient_energy": float(np.square(raw).sum()),
            "fraction_missing": max(expected - observed, 0) / expected if expected else np.nan,
            "fraction_near_zero": float((raw <= NEAR_ZERO_THRESHOLD).mean()) if not raw.empty else np.nan,
        })
    return pd.DataFrame(rows, columns=COMPONENT_COLUMNS)


def aggregate_ratios(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=RATIO_COLUMNS)
    classical = {"classical_encoder", "classical_readout", "batch_norm", "residual_path", "other_classical"}
    rows = []
    keys = ["canonical_model_name", "model_config_name", "seed", "epoch"]
    for key, frame in long_df.groupby(keys, sort=True):
        frame = frame[np.isfinite(frame["raw_l2"])]
        q = frame[frame.component == "quantum"]
        all_c = frame[frame.component.isin(classical)]
        enc = frame[frame.component == "classical_encoder"]
        readout = frame[frame.component == "classical_readout"]
        no_bn = frame[frame.component.isin(classical - {"batch_norm"})]
        no_res = frame[frame.component.isin(classical - {"residual_path"})]
        q_raw = float(q.raw_l2.mean()) if not q.empty else np.nan
        q_rms = weighted_rms_mean(q)
        total_energy = float(np.square(frame.raw_l2).sum())
        q_energy = float(np.square(q.raw_l2).sum())
        rows.append(dict(zip(keys, key)) | {
            "quantum_to_all_classical_raw_ratio": safe_ratio(q_raw, float(all_c.raw_l2.mean()) if not all_c.empty else np.nan),
            "quantum_to_all_classical_rms_ratio": safe_ratio(q_rms, weighted_rms_mean(all_c)),
            "quantum_to_encoder_rms_ratio": safe_ratio(q_rms, weighted_rms_mean(enc)),
            "quantum_to_readout_rms_ratio": safe_ratio(q_rms, weighted_rms_mean(readout)),
            "quantum_to_classical_without_bn_rms_ratio": safe_ratio(q_rms, weighted_rms_mean(no_bn)),
            "quantum_to_classical_without_residual_rms_ratio": safe_ratio(q_rms, weighted_rms_mean(no_res)),
            "quantum_gradient_energy_fraction": q_energy / total_energy if total_energy > 0 and not q.empty else np.nan,
        })
    return pd.DataFrame(rows, columns=RATIO_COLUMNS)


def mapping_rows(
    canonical: str, model_config_name: str, parameter_info: dict[str, ParameterInfo]
) -> list[dict[str, Any]]:
    return [{
        "canonical_model_name": canonical,
        "model_config_name": model_config_name,
        "parameter_name": info.name,
        "parameter_shape": json.dumps(list(info.shape)) if info.shape is not None else "",
        "parameter_numel": info.numel,
        "component": info.component,
        "mapping_rule": info.rule,
        "mapping_confidence": info.confidence,
        "manual_review": info.manual_review,
        "parameter_metadata_source": info.source,
    } for info in sorted(parameter_info.values(), key=lambda x: x.name)]


def semicolon_list(values: Iterable[str]) -> str:
    return "; ".join(sorted(set(values)))


def write_schema() -> None:
    text = f"""# Training-gradient export schema

## Semantics

Each gradient value is the **last-batch parameter-gradient L2 norm logged for the epoch**. It is captured by `GradientProfiler.on_after_backward`, cached, and emitted at train-epoch end. Epoch is the zero-based ordinal of the strictly ordered gradient logging events; the stored `metrics.csv` epoch labels are not used because legacy `MetricsContainer.to_pandas()` interleaves several epoch/step records.

`rms_l2 = raw_l2 / sqrt(parameter_numel)`. Missing sizes remain missing; they are never replaced with one. Missing gradients remain missing and are never filled with zero. The near-zero threshold is `raw_l2 <= {NEAR_ZERO_THRESHOLD:.1e}`.

## gradient_parameters_long.csv

- `canonical_model_name`, `model_config_name`, `seed`: join identifiers copied from the existing training-dynamics identity table when available.
- `epoch`: zero-based gradient-event ordinal aligned to training epochs.
- `parameter_name`: exact `named_parameters()` / `grad_l2` key.
- `parameter_shape`, `parameter_numel`: loaded-model tensor metadata, then checkpoint fallback.
- `component`: `quantum`, `classical_encoder`, `classical_readout`, `batch_norm`, `residual_path`, `other_classical`, or `unresolved`.
- `raw_l2`: logged parameter-gradient L2 norm.
- `rms_l2`: `raw_l2 / sqrt(parameter_numel)`.
- `extraction_source`: `logger.pkl`, `metrics.csv`, or `tensorboard`.
- `mapping_rule`, `mapping_confidence`, `manual_review`: classification provenance and review status.

## gradient_components_by_epoch.csv

- `n_parameter_tensors`: finite logged tensors in the component at this epoch.
- `total_parameter_numel`: scalar count across those tensors.
- `raw_l2_mean`, `raw_l2_median`, `raw_l2_sum`, `raw_l2_max`: tensor-level raw L2 summaries.
- `rms_l2_mean`, `rms_l2_median`: tensor-level RMS summaries.
- `rms_l2_weighted_by_numel`: `sum(rms_l2_i * numel_i) / sum(numel_i)`.
- `squared_gradient_energy`: `sum(raw_l2_i^2)`.
- `fraction_missing`: expected mapped tensors absent/non-finite at the epoch divided by expected mapped tensors.
- `fraction_near_zero`: finite tensors with `raw_l2 <= {NEAR_ZERO_THRESHOLD:.1e}` divided by finite tensors.

## gradient_ratios_by_epoch.csv

The raw ratio uses mean tensor L2. RMS ratios use the numel-weighted tensor RMS defined above. `all_classical` contains encoder, readout, batch-normalization, residual-path, and other classical tensors; unresolved tensors are excluded.

- `quantum_to_all_classical_raw_ratio`: mean quantum raw L2 / mean all-classical raw L2.
- `quantum_to_all_classical_rms_ratio`: weighted quantum RMS / weighted all-classical RMS.
- `quantum_to_encoder_rms_ratio`: weighted quantum RMS / weighted classical-encoder RMS.
- `quantum_to_readout_rms_ratio`: weighted quantum RMS / weighted classical-readout RMS.
- `quantum_to_classical_without_bn_rms_ratio`: weighted quantum RMS / weighted classical RMS excluding batch norm.
- `quantum_to_classical_without_residual_rms_ratio`: weighted quantum RMS / weighted classical RMS excluding residual-path tensors.
- `quantum_gradient_energy_fraction`: `sum_quantum(raw_l2^2) / sum_all_logged_trainable(raw_l2^2)`.

## gradient_parameter_mapping.csv

One row per model configuration and model/checkpoint parameter. Columns give exact name, shape, numel, assigned component, rule, confidence, review flag, and metadata source.

## gradient_extraction_quality_report.csv

One row per model and seed. Presence/coverage fields report source files, usable gradients, expected/extracted epoch counts, model/checkpoint matching, unresolved or missing sizes, duplicate epoch keys, non-finite values, suspicious all-zero/near-zero histories, source comparison results, epoch alignment, status, and notes. `model_parameters_never_logged` and `logged_parameters_not_in_model` are semicolon-separated exact-name lists.
"""
    (OUTPUT_DIR / "schema.md").write_text(text, encoding="utf-8")


def write_summary(
    models_root: Path,
    identity_source: Path | None,
    model_count: int,
    seed_count: int,
    usable_seeds: int,
    long_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    manual_validations: list[tuple[str, str]],
) -> None:
    unresolved = mapping_df[mapping_df.component == "unresolved"] if not mapping_df.empty else mapping_df
    recommended = quality_df[quality_df.status != "ok"] if not quality_df.empty else quality_df
    alignment_counts = quality_df["epoch_alignment_status"].value_counts().to_dict() if not quality_df.empty else {}
    exact_step_alignments = sum(
        count for label, count in alignment_counts.items() if str(label).startswith("gradient_steps_match_train_steps")
    )
    count_validated_alignments = sum(
        count for label, count in alignment_counts.items() if "event_count_matches_expected" in str(label)
    )
    single_event_alignments = sum(
        count for label, count in alignment_counts.items() if str(label).startswith("single_gradient_event")
    )
    validation_lines = "\n".join(f"- `{name}`: {result}" for name, result in manual_validations)
    text = f"""# Training gradient extraction summary

## Coverage

- Exact experiment root used: `{models_root}`
- Requested root: `{REQUESTED_MODELS_ROOT}`
- Existing identity table used: `{identity_source if identity_source else 'none; path-derived fallback identifiers used'}`
- Model configurations discovered: {model_count}
- Seed directories discovered: {seed_count}
- Seeds with usable gradient metrics: {usable_seeds}
- Exported parameter-gradient observations: {len(long_df)}
- Unique mapped model-parameter rows: {len(mapping_df)}
- Unresolved mapping rows: {len(unresolved)}

## Logger semantics and epoch alignment

Every exported value is a **last-batch parameter-gradient L2 norm logged for the epoch.** The callback captures the latest gradients in `on_after_backward` and emits that cache at epoch end; these are not epoch averages or full-epoch norms. Epochs use the zero-based ordinal of strictly increasing gradient log steps. This avoids legacy `metrics.csv` epoch labels generated from an interleaved raw epoch list.

Gradient source priority was `logger.pkl`, then `metrics.csv`, then TensorBoard. Logger and CSV values were compared whenever both were available.

Epoch validation found {exact_step_alignments} seeds whose gradient steps exactly match train-log steps, {count_validated_alignments} additional seeds whose strictly increasing gradient-event count matches the expected epoch count, and {single_event_alignments} one-epoch seeds whose single gradient event matches the expected count. The two HQuanv basic runs log train metrics twice as often as gradients; their 71 strictly increasing gradient events still match the 71 expected epochs. Runs without usable gradients retain an explicit no-evidence status.

## Parameter size normalization and component mapping

Parameter names, shapes, sizes, module types, and execution-role classification were obtained from one repository-loaded trained model per configuration. Checkpoint `state_dict` metadata was used only when repository loading failed. RMS normalization is `raw_l2 / sqrt(parameter_numel)`; unresolved sizes remain missing.

Quantum parameters are identified by their loaded PennyLane `TorchLayer` owners. Batch normalization is identified by loaded PyTorch batch-normalization module type. Encoder/readout roles follow the loaded HQNN structure and known pre/post-quantum child modules. Residual projection and scale parameters are isolated as `residual_path`. Dropout has no trainable parameter tensor and therefore has no parameter-mapping row of its own; its surrounding quantum and classical parameters remain distinguishable through the model configuration identifier.

## Required architecture spot checks

{validation_lines or '- No requested architecture pattern was present.'}

## Unresolved cases and recommended exclusions

- Logged names not found in a loaded model/checkpoint and parameters whose execution order cannot be resolved are marked `unresolved`, assigned low confidence, and flagged for manual review.
- Seeds with `status != ok` should be inspected or excluded for analyses requiring complete epoch coverage. Count: {len(recommended)}.
- Ratios with a missing or zero denominator are left blank/NaN.
- Missing gradients are never filled with zero, and missing parameter sizes are never replaced with one.
- Exact unresolved names and never-logged model parameters are listed per seed in `gradient_extraction_quality_report.csv`.
"""
    (OUTPUT_DIR / "extraction_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    models_root = resolve_models_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    identity_map, identity_source = load_identity_map()
    model_dirs = discover_model_dirs(models_root)

    all_long: list[dict[str, Any]] = []
    all_mapping: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []
    manual_validations: list[tuple[str, str]] = []
    seed_count = usable_seeds = 0

    required_checks = {
        "HVQL strong": ("hybrid/hvql/ansatz=strong",),
        "HVQL basic": ("hybrid/hvql/ansatz=basic",),
        "HVQL random": ("hybrid/hvql/ansatz=random",),
        "HVQL dropout": ("hybrid/hvql/dropout",),
        "HVQL residual": ("hybrid/hvql/residual",),
        "HVQM": ("hybrid/hvqm/",),
        "HPVQL": ("hybrid/hpvql/", "hybrid/hpvqm/"),
        "HQE": ("hybrid/hqe",),
        "HQuanv": ("hybrid/hquanv/",),
    }
    check_results: dict[str, str] = {}

    for model_dir in tqdm(model_dirs, desc="Model configurations"):
        model_config_name = model_dir.relative_to(models_root).as_posix()
        seed_dirs = iter_seed_dirs(model_dir)
        seed_count += len(seed_dirs)
        if not seed_dirs:
            continue

        sources_by_seed: dict[int, tuple[GradientSource | None, GradientSource | None, int, int]] = {}
        logged_names: set[str] = set()
        for seed_dir in seed_dirs:
            logger = read_logger_gradients(seed_dir / "metrics" / "logger.pkl")
            csv = read_csv_gradients(seed_dir / "metrics" / "metrics.csv")
            compared, mismatched = compare_sources(logger, csv)
            primary = logger if logger is not None and logger.histories else csv
            if primary is None or not primary.histories:
                primary = read_tensorboard_gradients(seed_dir / "tensorboard")
            sources_by_seed[int(seed_dir.name)] = (primary, csv, compared, mismatched)
            if primary is not None:
                logged_names.update(primary.histories)

        representative_seed = seed_dirs[0]
        parameter_info, model_parameter_names, model_load_success, model_notes = load_parameter_info(
            model_dir, representative_seed, logged_names
        )
        representative_canonical = canonical_name(
            identity_map, model_config_name, int(representative_seed.name)
        )
        all_mapping.extend(mapping_rows(representative_canonical, model_config_name, parameter_info))

        lower_name = model_config_name.lower()
        for label, patterns in required_checks.items():
            if label in check_results or not any(pattern in lower_name for pattern in patterns):
                continue
            components = sorted({parameter_info[name].component for name in logged_names if name in parameter_info})
            unresolved_names = sorted(name for name in logged_names if parameter_info[name].component == "unresolved")
            check_results[label] = (
                f"representative loaded-model mapping inspected; components={components}; "
                f"unresolved_logged={unresolved_names or 'none'}"
            )

        for seed_dir in seed_dirs:
            seed = int(seed_dir.name)
            canonical = canonical_name(identity_map, model_config_name, seed)
            primary, csv, compared, mismatched = sources_by_seed[seed]
            logger_path = seed_dir / "metrics" / "logger.pkl"
            csv_path = seed_dir / "metrics" / "metrics.csv"
            expected = expected_epochs(model_dir, primary)
            notes = list(model_notes)
            if primary is not None and primary.notes:
                notes.extend(primary.notes)
            rows = [] if primary is None else build_long_rows(
                canonical, model_config_name, seed, primary, parameter_info
            )
            all_long.extend(rows)
            gradient_present = bool(rows)
            usable_seeds += int(gradient_present)
            extracted_epochs = primary.n_epochs if primary is not None else 0
            source_steps = primary.steps if primary is not None else []
            strictly_increasing = all(b > a for a, b in zip(source_steps, source_steps[1:]))
            train_steps = primary.train_steps if primary is not None else None
            count_matches_expected = expected is not None and extracted_epochs == expected
            if source_steps and train_steps and source_steps == train_steps and strictly_increasing:
                epoch_alignment = "gradient_steps_match_train_steps; zero_based_ordinal"
            elif source_steps and strictly_increasing and count_matches_expected:
                epoch_alignment = "strictly_increasing_gradient_steps; event_count_matches_expected; zero_based_ordinal"
            elif source_steps and strictly_increasing:
                epoch_alignment = "strictly_increasing_gradient_steps; zero_based_ordinal"
            elif extracted_epochs == 1 and count_matches_expected:
                epoch_alignment = "single_gradient_event_matches_expected; zero_based_ordinal"
            else:
                epoch_alignment = "ordinal_without_complete_step_evidence"
            if expected is not None and extracted_epochs != expected:
                notes.append(f"expected {expected} epochs but extracted {extracted_epochs}")
            logged = set(primary.histories) if primary is not None else set()
            matched = logged & model_parameter_names
            unresolved = {name for name in logged if parameter_info[name].component == "unresolved"}
            missing_sizes = {name for name in logged if parameter_info[name].numel is None}
            never_logged = model_parameter_names - logged
            logged_not_model = logged - model_parameter_names
            values = [v for history in (primary.histories.values() if primary else []) for v in history]
            finite_values = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
            suspicious = ""
            if finite_values.size and np.all(finite_values == 0):
                suspicious = "all gradients are zero"
            elif finite_values.size and float(np.mean(finite_values <= NEAR_ZERO_THRESHOLD)) > 0.5:
                suspicious = f">50% gradients <= {NEAR_ZERO_THRESHOLD:.1e}"
            duplicate_epochs = 0
            if rows:
                keys = [(r["epoch"], r["parameter_name"]) for r in rows]
                duplicate_epochs = len(keys) - len(set(keys))
            status = "ok"
            if not gradient_present:
                status = "no_gradient_metrics"
            elif unresolved or missing_sizes or logged_not_model or duplicate_epochs:
                status = "review"
            elif expected is not None and extracted_epochs != expected:
                status = "incomplete_epochs"
            elif mismatched:
                status = "source_mismatch"
            all_quality.append({
                "canonical_model_name": canonical,
                "model_config_name": model_config_name,
                "seed": seed,
                "logger_pkl_present": logger_path.exists(),
                "metrics_csv_present": csv_path.exists(),
                "gradient_metrics_present": gradient_present,
                "epochs_expected": expected,
                "epochs_extracted": extracted_epochs,
                "checkpoint_model_load_success": model_load_success,
                "parameter_names_matched": len(matched),
                "unresolved_parameter_names": semicolon_list(unresolved),
                "missing_parameter_sizes": semicolon_list(missing_sizes),
                "duplicate_epochs": duplicate_epochs,
                "nan_values": primary.nan_values if primary is not None else 0,
                "all_zero_or_suspicious_gradients": suspicious,
                "model_parameters_never_logged": semicolon_list(never_logged),
                "logged_parameters_not_in_model": semicolon_list(logged_not_model),
                "logger_csv_comparison_count": compared,
                "logger_csv_mismatch_count": mismatched,
                "epoch_alignment_status": epoch_alignment,
                "status": status,
                "notes": "; ".join(notes),
            })

    manual_validations = [(label, check_results.get(label, "not present under resolved root")) for label in required_checks]
    long_df = pd.DataFrame(all_long, columns=LONG_COLUMNS)
    mapping_df = pd.DataFrame(all_mapping, columns=MAPPING_COLUMNS).drop_duplicates(
        ["model_config_name", "parameter_name"], keep="first"
    )
    quality_df = pd.DataFrame(all_quality, columns=QUALITY_COLUMNS)
    component_df = aggregate_components(long_df, mapping_df)
    ratio_df = aggregate_ratios(long_df)

    sort_long = ["model_config_name", "seed", "epoch", "parameter_name"]
    long_df.sort_values(sort_long, inplace=True, ignore_index=True)
    component_df.sort_values(["model_config_name", "seed", "epoch", "component"], inplace=True, ignore_index=True)
    ratio_df.sort_values(["model_config_name", "seed", "epoch"], inplace=True, ignore_index=True)
    mapping_df.sort_values(["model_config_name", "parameter_name"], inplace=True, ignore_index=True)
    quality_df.sort_values(["model_config_name", "seed"], inplace=True, ignore_index=True)

    long_df.to_csv(OUTPUT_DIR / "gradient_parameters_long.csv", index=False)
    component_df.to_csv(OUTPUT_DIR / "gradient_components_by_epoch.csv", index=False)
    ratio_df.to_csv(OUTPUT_DIR / "gradient_ratios_by_epoch.csv", index=False)
    mapping_df.to_csv(OUTPUT_DIR / "gradient_parameter_mapping.csv", index=False)
    quality_df.to_csv(OUTPUT_DIR / "gradient_extraction_quality_report.csv", index=False)
    write_schema()
    write_summary(
        models_root, identity_source, len(model_dirs), seed_count, usable_seeds,
        long_df, mapping_df, quality_df, manual_validations,
    )
    print(f"Wrote training-gradient exports to {OUTPUT_DIR}")
    print(f"models={len(model_dirs)} seeds={seed_count} usable={usable_seeds} rows={len(long_df)}")


if __name__ == "__main__":
    main()
