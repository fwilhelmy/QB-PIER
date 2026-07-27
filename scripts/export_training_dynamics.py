"""Export lightweight training-dynamics tables from saved experiment artifacts.

This script crawls a model artifact tree using `.model` anchor files, parses
metadata and metrics already written by training runs, and emits CSV/Markdown
outputs for downstream analysis. It does not train models, run attacks, import
model code, or load checkpoint tensors.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in lean analysis envs.
    yaml = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this script.

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else []


log = logging.getLogger(__name__)


OUTPUT_FILENAMES = [
    "training_curves_long.csv",
    "training_curves_wide.csv",
    "model_static_metadata.csv",
    "quantum_static_metadata.csv",
    "extraction_quality_report.csv",
    "extraction_summary.md",
]

LONG_COLUMNS = [
    "model_config_name",
    "canonical_model_name",
    "family",
    "architecture",
    "variant_configuration",
    "seed",
    "epoch",
    "step",
    "metric_name",
    "raw_metric_name",
    "metric_value",
    "source_file",
    "split",
    "metric_type",
]

WIDE_BASE_COLUMNS = [
    "model_config_name",
    "canonical_model_name",
    "family",
    "architecture",
    "variant_configuration",
    "seed",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "gradient_norm_mean",
    "gradient_norm_std",
]

STATIC_COLUMNS = [
    "model_config_name",
    "canonical_model_name",
    "model_class",
    "family",
    "architecture",
    "variant_configuration",
    "seed",
    "params_total",
    "params_classical",
    "params_quantum",
    "param_ratio_q_classical",
    "qubits",
    "epochs_configured",
    "epochs_completed",
    "training_time_seconds",
    "time_per_epoch",
    "device",
    "dataset",
    "optimizer",
    "learning_rate",
    "batch_size",
    "loss_function",
    "date",
    "hydra_overrides",
    "has_final_ckpt",
    "n_checkpoints",
    "best_checkpoint_metric",
    "best_checkpoint_epoch",
    "best_checkpoint_metric_name",
]

QUANTUM_COLUMNS = [
    "model_config_name",
    "canonical_model_name",
    "family",
    "architecture",
    "variant_configuration",
    "embedding_type",
    "ansatz_type",
    "number_of_qubits",
    "number_of_layers",
    "circuit_depth",
    "number_of_gates",
    "weights_shape",
    "qblocks",
    "depth_split_ratio",
    "gate_split_ratio",
    "params_quantum",
    "params_classical",
    "params_total",
    "param_ratio_q_classical",
    "readout_type",
    "measurement_observables",
    "source_file",
]

QUALITY_COLUMNS = [
    "model_config_name",
    "seed",
    "status",
    "missing_files",
    "missing_required_metrics",
    "missing_optional_metrics",
    "n_epochs_found",
    "first_epoch",
    "last_epoch",
    "has_nan",
    "duplicate_epochs",
    "notes",
]

REQUIRED_METRICS = ["train_loss", "train_acc", "val_loss", "val_acc"]
OPTIONAL_METRICS = ["test_loss", "test_acc", "gradient_norm_mean", "gradient_norm_std", "device_stat"]

FAMILY_MAP = {
    "classical": "Classical",
    "quantum": "Quantum",
    "hybrid": "Hybrid",
}

ARCHITECTURE_MAP = {
    "cnn": "CNN",
    "convnet": "ConvNet",
    "lenet": "LeNet",
    "mlp": "MLP",
    "mlpmixer": "MLPMixer",
    "resnet18": "ResNet18",
    "vqm": "VQM",
    "ampvqc": "VQM",
    "vqc": "VQC",
    "hvql": "HVQL",
    "hvqm": "HVQM",
    "hpvql": "HPVQL",
    "hpvqm": "HPVQM",
    "hquanv": "HQuanv",
    "hqe": "HQE",
    "drophvql": "DropHVQL",
    "dropvql": "DropHVQL",
    "reshvql": "ResHVQL",
    "resvql": "ResHVQL",
    "hqnn": "HQNN",
}

NON_METRIC_COLUMNS = {
    "epoch",
    "step",
    "global_step",
    "batch",
    "val_step",
    "valid_step",
    "validation_step",
    "test_step",
    "train_step",
    "grad_l2_step",
    "gradient_step",
    "performance_step",
    "device_step",
}

QUANTUM_TOKENS = (
    "quantum",
    "qnode",
    "qlayer",
    "torchlayer",
    "vqc",
    "vql",
    "quanv",
    "ansatz",
    "pennylane",
)


@dataclass
class ModelIdentity:
    model_config_name: str
    canonical_model_name: str
    family: str
    architecture: str
    variant_configuration: str
    model_class: str


@dataclass
class SeedExtraction:
    long_rows: list[dict[str, Any]] = field(default_factory=list)
    static_row: dict[str, Any] = field(default_factory=dict)
    quality_row: dict[str, Any] = field(default_factory=dict)
    metric_names: set[str] = field(default_factory=set)
    raw_metric_map: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    malformed: bool = False


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    except OSError as exc:
        log.warning("Failed to read text file %s: %s", path, exc)
        return ""


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Failed to read JSON %s: %s", path, exc)
        return None


def read_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if yaml is not None:
            return yaml.safe_load(text) or {}
        return simple_yaml_load(text)
    except Exception as exc:
        log.warning("Failed to read YAML %s: %s", path, exc)
        return {}


def parse_yaml_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith("[") or text.startswith("{"):
        try:
            return ast.literal_eval(text)
        except Exception:
            return text
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", text) or re.fullmatch(
            r"[-+]?\d+[eE][-+]?\d+", text
        ):
            return float(text)
    except ValueError:
        return text
    return text


def simple_yaml_load(text: str) -> Any:
    """Small fallback parser for simple Hydra-style YAML maps and lists."""
    parsed_lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        parsed_lines.append((indent, raw_line.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(parsed_lines):
            return {}, index
        is_list = parsed_lines[index][1].startswith("- ")
        if is_list:
            values = []
            while index < len(parsed_lines):
                line_indent, stripped = parsed_lines[index]
                if line_indent < indent or not stripped.startswith("- "):
                    break
                if line_indent > indent:
                    child, index = parse_block(index, line_indent)
                    values.append(child)
                    continue
                content = stripped[2:].strip()
                if content == "":
                    child, index = parse_block(index + 1, parsed_lines[index + 1][0])
                    values.append(child)
                elif ":" in content and not content.startswith(("http://", "https://")):
                    key, _, rest = content.partition(":")
                    item = {key.strip(): parse_yaml_scalar(rest.strip()) if rest.strip() else None}
                    index += 1
                    values.append(item)
                else:
                    values.append(parse_yaml_scalar(content))
                    index += 1
            return values, index

        values: dict[str, Any] = {}
        while index < len(parsed_lines):
            line_indent, stripped = parsed_lines[index]
            if line_indent < indent or stripped.startswith("- "):
                break
            if line_indent > indent:
                index += 1
                continue
            key, sep, rest = stripped.partition(":")
            if not sep:
                index += 1
                continue
            key = key.strip()
            rest = rest.strip()
            if rest:
                values[key] = parse_yaml_scalar(rest)
                index += 1
            elif index + 1 < len(parsed_lines) and parsed_lines[index + 1][0] > line_indent:
                child, index = parse_block(index + 1, parsed_lines[index + 1][0])
                values[key] = child
            else:
                values[key] = None
                index += 1
        return values, index

    if not parsed_lines:
        return {}
    result, _ = parse_block(0, parsed_lines[0][0])
    return result


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def safe_get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            return default
    return cur


def first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def compact_json(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def normalize_family(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    return FAMILY_MAP.get(text.lower(), text[:1].upper() + text[1:])


def normalize_architecture(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    lower = text.lower().replace("-", "").replace("_", "")
    if re.fullmatch(r"vqm\d+", lower):
        return text.upper()
    return ARCHITECTURE_MAP.get(lower, ARCHITECTURE_MAP.get(text.lower(), text))


def safe_column_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[/.\-\s:]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def find_model_dirs(models_root: Path) -> list[Path]:
    anchors = sorted(p for p in models_root.rglob(".model") if p.is_file())
    return [anchor.parent for anchor in anchors]


def iter_seed_dirs(model_dir: Path) -> list[Path]:
    return sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def load_hydra_config(model_dir: Path, seed_dir: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    model_config_path = model_dir / ".hydra" / "config.yaml"
    seed_config_path = seed_dir / ".hydra" / "config.yaml" if seed_dir is not None else None
    model_cfg = read_yaml(model_config_path)
    seed_cfg = read_yaml(seed_config_path) if seed_config_path is not None else {}
    cfg = merge_dicts(model_cfg if isinstance(model_cfg, dict) else {}, seed_cfg if isinstance(seed_cfg, dict) else {})
    source = seed_config_path if seed_config_path is not None and seed_config_path.exists() else None
    if source is None and model_config_path.exists():
        source = model_config_path
    return cfg, source


def load_overrides(model_dir: Path, seed_dir: Path | None = None) -> tuple[Any, Path | None]:
    candidates = []
    if seed_dir is not None:
        candidates.append(seed_dir / ".hydra" / "overrides.yaml")
    candidates.append(model_dir / ".hydra" / "overrides.yaml")
    for path in candidates:
        if path.exists():
            return read_yaml(path), path
    return [], None


def load_hparams(seed_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
    if seed_dir is None:
        return {}, None
    path = seed_dir / "tensorboard" / "hparams.yaml"
    payload = read_yaml(path)
    return payload if isinstance(payload, dict) else {}, path if path.exists() else None


def path_metadata(models_root: Path, model_dir: Path) -> dict[str, str]:
    rel = model_dir.relative_to(models_root)
    parts = list(rel.parts)
    lower_parts = [p.lower() for p in parts]

    family_idx = None
    family = "Unknown"
    for idx, part in enumerate(lower_parts):
        if part in FAMILY_MAP:
            family_idx = idx
            family = FAMILY_MAP[part]
            break
    if family_idx is None:
        for idx, part in enumerate(lower_parts):
            for token, label in FAMILY_MAP.items():
                if token in part:
                    family_idx = idx
                    family = label
                    break
            if family_idx is not None:
                break

    if family_idx is not None and family_idx + 1 < len(parts):
        architecture = parts[family_idx + 1]
        variant_parts = parts[family_idx + 2 :]
    elif len(parts) >= 2 and ("=" in parts[-1] or "," in parts[-1]):
        architecture = parts[-2]
        variant_parts = [parts[-1]]
    else:
        architecture = parts[-1] if parts else model_dir.name
        variant_parts = []

    return {
        "family": family,
        "architecture": normalize_architecture(architecture),
        "variant": "/".join(variant_parts),
    }


def find_first_key(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys and value not in (None, ""):
                return value
        for value in obj.values():
            found = find_first_key(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_first_key(value, keys)
            if found not in (None, ""):
                return found
    return None


def format_knob_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return compact_json(value)


def collect_variant_knobs(cfg: dict[str, Any], hparams: dict[str, Any], path_variant: str) -> str:
    model_cfg = safe_get(cfg, "model", default={}) or {}
    knobs: list[tuple[str, Any]] = []

    candidate_paths = [
        ("n_qdepth", ("n_qdepth",)),
        ("qdepth", ("qdepth",)),
        ("n_layers", ("n_layers",)),
        ("n_qubits", ("n_qubits",)),
        ("qubits", ("qubits",)),
        ("qblocks", ("qblocks",)),
        ("hidden_sizes", ("hidden_sizes",)),
        ("conv_channels", ("conv_channels",)),
        ("fc_sizes", ("fc_sizes",)),
        ("dropout", ("dropout",)),
        ("dropout_p", ("dropout_p",)),
        ("residual", ("residual",)),
        ("ratio_imprim", ("ratio_imprim",)),
        ("depth_split_ratio", ("depth_split_ratio",)),
        ("gate_split_ratio", ("gate_split_ratio",)),
        ("ansatz", ("ansatz", "class")),
        ("embedding", ("embedding",)),
        ("rotation", ("rotation",)),
        ("patch_size", ("patch_size",)),
        ("stride", ("stride",)),
        ("encoder", ("encoder", "name")),
        ("classifier", ("classifier", "name")),
        ("encoder_n_qdepth", ("encoder", "params", "n_qdepth")),
        ("encoder_embedding", ("encoder", "params", "embedding")),
        ("encoder_rotation", ("encoder", "params", "rotation")),
        ("encoder_patch_size", ("encoder", "params", "patch_size")),
        ("encoder_stride", ("encoder", "params", "stride")),
        ("encoder_ansatz", ("encoder", "params", "ansatz", "class")),
        ("classifier_hidden_sizes", ("classifier", "params", "hidden_sizes")),
    ]
    for label, path in candidate_paths:
        value = safe_get(model_cfg, *path)
        if value not in (None, "", []):
            knobs.append((label, value))

    for label in ["n_qdepth", "qdepth", "n_layers", "n_qubits", "qblocks"]:
        value = hparams.get(label)
        if value not in (None, "", []) and (label, value) not in knobs:
            knobs.append((label, value))

    seen = set()
    formatted = []
    for key, value in knobs:
        token = f"{key}={format_knob_value(value)}"
        if token not in seen:
            seen.add(token)
            formatted.append(token)

    if formatted:
        return ", ".join(formatted)
    return path_variant or "baseline configuration"


def build_identity(
    models_root: Path,
    model_dir: Path,
    cfg: dict[str, Any],
    hparams: dict[str, Any],
) -> ModelIdentity:
    rel = model_dir.relative_to(models_root).as_posix()
    path_meta = path_metadata(models_root, model_dir)
    model_cfg = safe_get(cfg, "model", default={}) or {}

    explicit_family = first_present(
        model_cfg.get("family"),
        model_cfg.get("model_family"),
        model_cfg.get("group"),
    )
    path_family = None if path_meta["family"] == "Unknown" else path_meta["family"]
    family = normalize_family(explicit_family or path_family or model_cfg.get("model_type"))

    explicit_architecture = first_present(
        model_cfg.get("architecture"),
        model_cfg.get("name"),
        model_cfg.get("model_name"),
    )
    path_architecture = None if path_meta["architecture"] == "Unknown" else path_meta["architecture"]
    architecture = normalize_architecture(explicit_architecture or path_architecture or model_cfg.get("model_class"))

    model_class = read_text(model_dir / ".model").strip()
    model_class = str(first_present(model_class, model_cfg.get("model_class"), architecture) or "")

    variant = collect_variant_knobs(cfg, hparams, path_meta["variant"])
    canonical = f"{family}/{architecture}"
    if variant and variant != "baseline configuration":
        canonical = f"{canonical}/{variant}"

    return ModelIdentity(
        model_config_name=rel,
        canonical_model_name=canonical,
        family=family,
        architecture=architecture,
        variant_configuration=variant,
        model_class=model_class,
    )


def normalize_metric_name(raw_name: str) -> str:
    raw = str(raw_name).strip()
    lower = raw.lower()
    canonical = safe_column_name(lower.replace("validation", "val").replace("valid", "val").replace("accuracy", "acc"))
    aliases = {
        "train_loss": "train_loss",
        "train_acc": "train_acc",
        "val_loss": "val_loss",
        "val_acc": "val_acc",
        "test_loss": "test_loss",
        "test_acc": "test_acc",
    }
    if canonical in aliases:
        return aliases[canonical]
    if canonical in {"train_error", "training_error"}:
        return "train_loss"
    if canonical in {"val_error", "validation_error", "valid_error"}:
        return "val_loss"
    if canonical in {"test_accuracy"}:
        return "test_acc"
    return canonical


def classify_metric(metric_name: str, raw_name: str = "") -> tuple[str, str]:
    name = normalize_metric_name(metric_name)
    raw = safe_column_name(raw_name or metric_name)
    if name.startswith("train_"):
        split = "train"
    elif name.startswith("val_") or name.startswith("valid_") or name.startswith("validation_"):
        split = "val"
    elif name.startswith("test_"):
        split = "test"
    elif raw.startswith("grad_l2") or "gradient" in raw or raw.startswith("grad_"):
        split = "gradient"
    elif raw.startswith("performance") or raw.startswith("device") or "cpu" in raw or "mem_" in raw or "memory" in raw:
        split = "device"
    else:
        split = "other"

    if "loss" in name:
        metric_type = "loss"
    elif "acc" in name or "accuracy" in name:
        metric_type = "accuracy"
    elif split == "gradient":
        metric_type = "gradient_norm"
    elif split == "device":
        metric_type = "device_stat"
    else:
        metric_type = "other"
    return split, metric_type


def is_metric_column(column: str) -> bool:
    return safe_column_name(column) not in NON_METRIC_COLUMNS


def pick_epoch_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if safe_column_name(column) == "epoch":
            return column
    return None


def pick_step(row: pd.Series, raw_metric_name: str) -> Any:
    raw = safe_column_name(raw_metric_name)
    candidates: list[str] = []
    if raw.startswith("grad_l2") or raw.startswith("grad_") or "gradient" in raw:
        candidates.extend(["grad_l2_step", "gradient_step"])
    if raw.startswith("performance") or raw.startswith("device"):
        candidates.extend(["performance_step", "device_step"])
    if raw.startswith("val_"):
        candidates.extend(["val_step", "valid_step", "validation_step"])
    if raw.startswith("test_"):
        candidates.append("test_step")
    if raw.startswith("train_"):
        candidates.append("train_step")
    candidates.extend(["step", "global_step"])

    normalized_columns = {safe_column_name(col): col for col in row.index}
    for candidate in candidates:
        col = normalized_columns.get(candidate)
        if col is not None and pd.notna(row[col]):
            return row[col]
    return np.nan


def dataframe_from_json_payload(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict) and isinstance(payload.get("metrics"), (list, dict)):
        return dataframe_from_json_payload(payload["metrics"])
    if not isinstance(payload, dict):
        return pd.DataFrame()

    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}_{key}" if prefix else str(key)
                walk(child_prefix, child)
        else:
            flat[prefix] = value

    walk("", payload)
    series = {}
    for key, value in flat.items():
        if isinstance(value, list):
            series[key] = pd.Series(value)
        else:
            series[key] = pd.Series([value])
    return pd.DataFrame(series)


def dataframe_from_pickle(path: Path) -> pd.DataFrame:
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        log.warning("Failed to read logger pickle %s: %s", path, exc)
        return pd.DataFrame()

    if hasattr(payload, "to_pandas"):
        try:
            return payload.to_pandas()
        except Exception as exc:
            log.warning("logger.pkl to_pandas failed for %s: %s", path, exc)
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, (dict, list)):
        return dataframe_from_json_payload(payload)
    return pd.DataFrame()


def read_metrics(seed_dir: Path | None) -> tuple[pd.DataFrame, Path | None, list[str], bool]:
    if seed_dir is None:
        return pd.DataFrame(), None, ["no seed directory"], False

    notes: list[str] = []
    metrics_dir = seed_dir / "metrics"
    csv_path = metrics_dir / "metrics.csv"
    json_path = metrics_dir / "metrics.json"
    logger_path = metrics_dir / "logger.pkl"

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path), csv_path, notes, False
        except Exception as exc:
            notes.append(f"metrics.csv malformed: {exc}")
            log.warning("Malformed metrics.csv at %s: %s", csv_path, exc)

    if json_path.exists():
        df = dataframe_from_json_payload(read_json(json_path))
        if not df.empty:
            notes.append("metrics loaded from metrics.json fallback")
            return df, json_path, notes, False
        notes.append("metrics.json present but empty or unsupported")

    if logger_path.exists():
        df = dataframe_from_pickle(logger_path)
        if not df.empty:
            notes.append("metrics loaded from logger.pkl fallback")
            return df, logger_path, notes, False
        notes.append("logger.pkl present but could not be parsed")

    return pd.DataFrame(), None, notes, csv_path.exists()


def metrics_to_long_rows(
    df: pd.DataFrame,
    source_file: Path | None,
    identity: ModelIdentity,
    seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if df.empty:
        return [], {}

    epoch_col = pick_epoch_column(df)
    raw_metric_map: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        epoch = row[epoch_col] if epoch_col is not None and pd.notna(row[epoch_col]) else np.nan
        for column in df.columns:
            if not is_metric_column(column):
                continue
            value = as_float(row[column])
            if value is None:
                continue
            metric_name = normalize_metric_name(column)
            raw_metric_map[str(column)] = metric_name
            split, metric_type = classify_metric(metric_name, str(column))
            rows.append(
                {
                    "model_config_name": identity.model_config_name,
                    "canonical_model_name": identity.canonical_model_name,
                    "family": identity.family,
                    "architecture": identity.architecture,
                    "variant_configuration": identity.variant_configuration,
                    "seed": seed,
                    "epoch": epoch,
                    "step": pick_step(row, str(column)),
                    "metric_name": metric_name,
                    "raw_metric_name": str(column),
                    "metric_value": value,
                    "source_file": str(source_file) if source_file is not None else "",
                    "split": split,
                    "metric_type": metric_type,
                }
            )
    return rows, raw_metric_map


def parse_tensorboard_scalars(
    seed_dir: Path | None,
    identity: ModelIdentity,
    seed: int | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    if seed_dir is None:
        return [], [], {}
    tb_dir = seed_dir / "tensorboard"
    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return [], ["tensorboard events requested but no event files found"], {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        return [], [f"tensorboard events requested but tensorboard is unavailable: {exc}"], {}

    rows: list[dict[str, Any]] = []
    raw_metric_map: dict[str, str] = {}
    notes: list[str] = []
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
        except Exception as exc:
            notes.append(f"failed to parse tensorboard event {event_file.name}: {exc}")
            continue
        for tag in scalar_tags:
            metric_name = normalize_metric_name(tag)
            raw_metric_map[tag] = metric_name
            split, metric_type = classify_metric(metric_name, tag)
            for scalar in accumulator.Scalars(tag):
                rows.append(
                    {
                        "model_config_name": identity.model_config_name,
                        "canonical_model_name": identity.canonical_model_name,
                        "family": identity.family,
                        "architecture": identity.architecture,
                        "variant_configuration": identity.variant_configuration,
                        "seed": seed,
                        "epoch": np.nan,
                        "step": scalar.step,
                        "metric_name": metric_name,
                        "raw_metric_name": tag,
                        "metric_value": float(scalar.value),
                        "source_file": str(event_file),
                        "split": split,
                        "metric_type": metric_type,
                    }
                )
    return rows, notes, raw_metric_map


def _parse_number_token(token: str) -> int | None:
    token = token.strip().replace(",", "")
    if not token or token == "--" or not re.fullmatch(r"\d+", token):
        return None
    return int(token)


def parse_model_info(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "params_total": None,
        "params_classical": None,
        "params_quantum": None,
        "epochs_completed": None,
        "training_time_seconds": None,
        "time_per_epoch": None,
        "qubits_from_shapes": None,
    }
    if not text:
        return out

    patterns = {
        "params_total": r"Total params:\s*([\d,]+)",
        "epochs_completed": r"Trained epochs:\s*(\d+)",
        "time_per_epoch": r"Training time per epoch:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
        "training_time_seconds": r"Total training time(?:\s*\(per seed\))?:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "")
            out[key] = float(value)

    classical = 0
    quantum = 0
    saw_param_line = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or "Param #" in line or "===" in line or "Total params" in line:
            continue
        match = re.match(r"^(?P<layer>.+?)\s{2,}(?P<params>[\d,]+|--)\s{2,}.*$", line)
        if not match:
            continue
        params = _parse_number_token(match.group("params"))
        if params is None:
            continue
        saw_param_line = True
        layer_name = match.group("layer").lower()
        if any(token in layer_name for token in QUANTUM_TOKENS):
            quantum += params
        else:
            classical += params

    if saw_param_line:
        out["params_classical"] = float(classical)
        out["params_quantum"] = float(quantum)

    for line in text.splitlines():
        if "VQC:" not in line:
            continue
        shapes = re.findall(r"\[-1,\s*(\d+)\]", line)
        if shapes:
            out["qubits_from_shapes"] = int(shapes[-1])
            break

    return out


def parse_training_time_from_text(text: str) -> float | None:
    if not text:
        return None
    info = parse_model_info(text)
    if info["training_time_seconds"] is not None:
        return float(info["training_time_seconds"])
    hms = re.search(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if hms:
        hours, minutes, seconds = hms.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return None


def find_info_text(model_dir: Path, seed_dir: Path | None) -> tuple[str, Path | None]:
    candidates = []
    if seed_dir is not None:
        candidates.append(seed_dir / "model.info")
    candidates.append(model_dir / "model.info")
    for path in candidates:
        if path.exists():
            return read_text(path), path
    return "", None


def find_training_log_text(model_dir: Path, seed_dir: Path | None) -> tuple[str, Path | None]:
    candidates = []
    if seed_dir is not None:
        candidates.extend([seed_dir / "training.log", seed_dir / "experiment.log"])
    candidates.extend([model_dir / "training.log", model_dir / "experiment.log", model_dir / "main.log"])
    for path in candidates:
        if path.exists():
            return read_text(path), path
    return "", None


def performance_time(model_dir: Path, seed: int | None) -> float | None:
    payload = read_json(model_dir / "performance.json")
    if not isinstance(payload, dict):
        return None
    if seed is not None:
        seed_times = payload.get("seed_times")
        if isinstance(seed_times, dict) and str(seed) in seed_times:
            return as_float(seed_times[str(seed)])
    return as_float(payload.get("average"))


def checkpoint_summary(seed_dir: Path | None) -> tuple[bool, int, float | None, int | None, str]:
    if seed_dir is None:
        return False, 0, None, None, ""
    final_ckpt = seed_dir / "final.ckpt"
    ckpt_dir = seed_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    best_value: float | None = None
    best_epoch: int | None = None
    best_metric_name = ""
    best_sort = float("inf")
    for path in ckpts:
        epoch_match = re.search(r"epoch[=_-](\d+)", path.name, flags=re.IGNORECASE)
        metric_match = None
        for candidate in re.finditer(r"([A-Za-z][A-Za-z0-9_/-]*)[=_-]([0-9]+(?:\.[0-9]+)?)", path.stem):
            if candidate.group(1).lower() != "epoch":
                metric_match = candidate
                break
        if metric_match is not None:
            metric_name = metric_match.group(1)
            metric_value = float(metric_match.group(2))
            sort_value = metric_value if "loss" in metric_name.lower() else -metric_value
            if sort_value < best_sort:
                best_sort = sort_value
                best_value = metric_value
                best_metric_name = metric_name
                best_epoch = int(epoch_match.group(1)) if epoch_match else None
    return final_ckpt.exists(), len(ckpts), best_value, best_epoch, best_metric_name


def configured_epochs(cfg: dict[str, Any]) -> int | None:
    return as_int(
        first_present(
            safe_get(cfg, "training", "epochs"),
            safe_get(cfg, "trainer", "max_epochs"),
            safe_get(cfg, "fit", "max_epochs"),
        )
    )


def configured_device(cfg: dict[str, Any]) -> str:
    device_parts = []
    device_cfg = safe_get(cfg, "device", default={}) or {}
    qdevice_cfg = safe_get(cfg, "qdevice", default={}) or {}
    for key in ["device", "accelerator", "devices"]:
        value = device_cfg.get(key)
        if value not in (None, ""):
            device_parts.append(f"{key}={value}")
    qdevice = qdevice_cfg.get("QDEVICE") or qdevice_cfg.get("device")
    if qdevice not in (None, ""):
        device_parts.append(f"qdevice={qdevice}")
    return ", ".join(device_parts)


def configured_dataset(cfg: dict[str, Any]) -> str:
    return str(
        first_present(
            safe_get(cfg, "dataset", "dataset"),
            safe_get(cfg, "dataset", "name"),
            safe_get(cfg, "data", "dataset"),
        )
        or ""
    )


def configured_optimizer(cfg: dict[str, Any]) -> str:
    return str(
        first_present(
            safe_get(cfg, "training", "optimiser", "name"),
            safe_get(cfg, "training", "optimizer", "name"),
            safe_get(cfg, "optimizer", "name"),
        )
        or ""
    )


def configured_learning_rate(cfg: dict[str, Any]) -> float | None:
    return as_float(
        first_present(
            safe_get(cfg, "training", "optimiser", "learning_rate"),
            safe_get(cfg, "training", "optimizer", "learning_rate"),
            safe_get(cfg, "training", "learning_rate"),
            safe_get(cfg, "optimizer", "learning_rate"),
            safe_get(cfg, "training", "optimiser", "optimiser_config", "lr"),
        )
    )


def configured_batch_size(cfg: dict[str, Any]) -> int | None:
    return as_int(
        first_present(
            safe_get(cfg, "dataset", "train_batch_size"),
            safe_get(cfg, "dataset", "batch_size"),
            safe_get(cfg, "training", "batch_size"),
            safe_get(cfg, "data", "batch_size"),
        )
    )


def configured_loss(cfg: dict[str, Any]) -> str:
    return str(
        first_present(
            safe_get(cfg, "training", "loss_fn"),
            safe_get(cfg, "training", "loss_function"),
            safe_get(cfg, "loss", "name"),
        )
        or ""
    )


def infer_date(model_dir: Path) -> str:
    for part in model_dir.parts:
        match = re.search(r"(\d{4}-\d{2}-\d{2})(?:[@T](\d{2})[-:](\d{2})[-:](\d{2}))?", part)
        if match:
            date = match.group(1)
            if match.group(2):
                return f"{date}T{match.group(2)}:{match.group(3)}:{match.group(4)}"
            return date
    try:
        return datetime.fromtimestamp((model_dir / ".model").stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def infer_qubits(
    cfg: dict[str, Any],
    hparams: dict[str, Any],
    info: dict[str, Any],
    identity: ModelIdentity,
) -> int | None:
    model_cfg = safe_get(cfg, "model", default={}) or {}
    explicit = first_present(
        find_first_key(model_cfg, {"n_qubits", "qubits", "number_of_qubits", "wires"}),
        hparams.get("n_qubits"),
        hparams.get("qubits"),
        info.get("qubits_from_shapes"),
    )
    value = as_int(explicit)
    if value is not None:
        return value

    patch_size = as_int(first_present(
        safe_get(model_cfg, "encoder", "params", "patch_size"),
        safe_get(model_cfg, "patch_size"),
    ))
    if patch_size is not None:
        channels = safe_get(cfg, "dataset", "input_shape", default=[1])
        if isinstance(channels, list) and channels:
            return int(channels[0]) * patch_size * patch_size
        return patch_size * patch_size

    output_size = as_int(hparams.get("output_size"))
    if output_size is not None and identity.family == "Quantum":
        return output_size
    return None


def variant_value(variant: str, keys: set[str]) -> Any:
    if not variant:
        return None
    for key in keys:
        pattern = rf"(?:^|[,/]\s*){re.escape(key)}=([^,]+)"
        match = re.search(pattern, variant)
        if match:
            return parse_yaml_scalar(match.group(1).strip())
    return None


def infer_layers(cfg: dict[str, Any], hparams: dict[str, Any], variant: str = "") -> int | None:
    model_cfg = safe_get(cfg, "model", default={}) or {}
    return as_int(
        first_present(
            find_first_key(model_cfg, {"n_qdepth", "qdepth", "n_layers", "number_of_layers", "layers"}),
            hparams.get("n_qdepth"),
            hparams.get("qdepth"),
            hparams.get("n_layers"),
            variant_value(variant, {"n_qdepth", "qdepth", "n_layers", "number_of_layers", "layers", "encoder_n_qdepth"}),
        )
    )


def infer_quantum_field(cfg: dict[str, Any], hparams: dict[str, Any], keys: set[str]) -> Any:
    model_cfg = safe_get(cfg, "model", default={}) or {}
    return first_present(find_first_key(model_cfg, keys), find_first_key(hparams, keys))


def param_ratio(params_quantum: Any, params_classical: Any) -> float | None:
    q = as_float(params_quantum)
    c = as_float(params_classical)
    if q is None or c is None or c == 0:
        return None
    return q / c


def infer_param_split(info: dict[str, Any], identity: ModelIdentity) -> tuple[int | None, int | None, int | None]:
    total = as_int(info.get("params_total"))
    classical = as_int(info.get("params_classical"))
    quantum = as_int(info.get("params_quantum"))
    if total is not None and classical is None and quantum is None:
        if identity.family == "Classical":
            classical = total
            quantum = 0
        elif identity.family == "Quantum":
            classical = 0
            quantum = total
    if total is None and classical is not None and quantum is not None:
        total = classical + quantum
    return total, classical, quantum


def build_static_row(
    model_dir: Path,
    seed_dir: Path | None,
    identity: ModelIdentity,
    cfg: dict[str, Any],
    hparams: dict[str, Any],
    overrides: Any,
    metrics_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], Path | None, list[str]]:
    seed = int(seed_dir.name) if seed_dir is not None and seed_dir.name.isdigit() else None
    info_text, info_path = find_info_text(model_dir, seed_dir)
    log_text, _ = find_training_log_text(model_dir, seed_dir)
    info = parse_model_info(info_text)
    total, classical, quantum = infer_param_split(info, identity)

    epoch_col = pick_epoch_column(metrics_df)
    epochs_found = 0
    first_epoch = None
    last_epoch = None
    if epoch_col is not None and not metrics_df.empty:
        epochs = pd.to_numeric(metrics_df[epoch_col], errors="coerce").dropna()
        if not epochs.empty:
            epochs_found = int(epochs.nunique())
            first_epoch = int(epochs.min())
            last_epoch = int(epochs.max())

    epochs_completed = as_int(info.get("epochs_completed"))
    if epochs_completed is None and epochs_found:
        epochs_completed = epochs_found

    training_time = first_present(
        info.get("training_time_seconds"),
        performance_time(model_dir, seed),
        parse_training_time_from_text(log_text),
    )
    training_time = as_float(training_time)
    time_per_epoch = as_float(info.get("time_per_epoch"))
    if time_per_epoch is None and training_time is not None and epochs_completed:
        time_per_epoch = training_time / epochs_completed

    has_final_ckpt, n_checkpoints, best_metric, best_epoch, best_metric_name = checkpoint_summary(seed_dir)
    qubits = infer_qubits(cfg, hparams, info, identity)
    notes: list[str] = []
    if info_path is None:
        notes.append("model.info missing")

    row = {
        "model_config_name": identity.model_config_name,
        "canonical_model_name": identity.canonical_model_name,
        "model_class": identity.model_class,
        "family": identity.family,
        "architecture": identity.architecture,
        "variant_configuration": identity.variant_configuration,
        "seed": seed,
        "params_total": total,
        "params_classical": classical,
        "params_quantum": quantum,
        "param_ratio_q_classical": param_ratio(quantum, classical),
        "qubits": qubits,
        "epochs_configured": configured_epochs(cfg),
        "epochs_completed": epochs_completed,
        "training_time_seconds": training_time,
        "time_per_epoch": time_per_epoch,
        "device": configured_device(cfg),
        "dataset": configured_dataset(cfg),
        "optimizer": configured_optimizer(cfg),
        "learning_rate": configured_learning_rate(cfg),
        "batch_size": configured_batch_size(cfg),
        "loss_function": configured_loss(cfg),
        "date": infer_date(model_dir),
        "hydra_overrides": compact_json(overrides),
        "has_final_ckpt": has_final_ckpt,
        "n_checkpoints": n_checkpoints,
        "best_checkpoint_metric": best_metric,
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_metric_name": best_metric_name,
    }
    return row, info, info_path, notes


def build_quantum_row(
    identity: ModelIdentity,
    cfg: dict[str, Any],
    hparams: dict[str, Any],
    static_row: dict[str, Any],
    info_source: Path | None,
    config_source: Path | None,
) -> dict[str, Any] | None:
    is_quantumish = identity.family in {"Quantum", "Hybrid"} or any(
        token in identity.canonical_model_name.lower() for token in QUANTUM_TOKENS
    )
    if not is_quantumish:
        return None

    source_file = str(first_present(info_source, config_source) or "")
    ansatz = first_present(
        infer_quantum_field(cfg, hparams, {"ansatz_type", "ansatz", "class"}),
        variant_value(identity.variant_configuration, {"ansatz", "ansatz_type", "encoder_ansatz"}),
    )
    if isinstance(ansatz, dict):
        ansatz = ansatz.get("class") or compact_json(ansatz)

    row = {
        "model_config_name": identity.model_config_name,
        "canonical_model_name": identity.canonical_model_name,
        "family": identity.family,
        "architecture": identity.architecture,
        "variant_configuration": identity.variant_configuration,
        "embedding_type": first_present(
            infer_quantum_field(cfg, hparams, {"embedding", "embedding_type"}),
            variant_value(identity.variant_configuration, {"embedding", "embedding_type", "encoder_embedding"}),
        ),
        "ansatz_type": ansatz,
        "number_of_qubits": static_row.get("qubits"),
        "number_of_layers": infer_layers(cfg, hparams, identity.variant_configuration),
        "circuit_depth": infer_quantum_field(cfg, hparams, {"circuit_depth", "depth"}),
        "number_of_gates": infer_quantum_field(cfg, hparams, {"number_of_gates", "num_gates", "gates"}),
        "weights_shape": infer_quantum_field(cfg, hparams, {"weights_shape", "weight_shapes"}),
        "qblocks": first_present(
            infer_quantum_field(cfg, hparams, {"qblocks"}),
            variant_value(identity.variant_configuration, {"qblocks"}),
        ),
        "depth_split_ratio": first_present(
            infer_quantum_field(cfg, hparams, {"depth_split_ratio"}),
            variant_value(identity.variant_configuration, {"depth_split_ratio"}),
        ),
        "gate_split_ratio": first_present(
            infer_quantum_field(cfg, hparams, {"gate_split_ratio"}),
            variant_value(identity.variant_configuration, {"gate_split_ratio"}),
        ),
        "params_quantum": static_row.get("params_quantum"),
        "params_classical": static_row.get("params_classical"),
        "params_total": static_row.get("params_total"),
        "param_ratio_q_classical": static_row.get("param_ratio_q_classical"),
        "readout_type": infer_quantum_field(cfg, hparams, {"readout_type", "readout"}),
        "measurement_observables": infer_quantum_field(cfg, hparams, {"measurement_observables", "observables"}),
        "source_file": source_file,
    }
    return {key: clean_scalar(value) for key, value in row.items()}


def quality_from_metrics(
    identity: ModelIdentity,
    seed: int | None,
    metrics_df: pd.DataFrame,
    long_rows: list[dict[str, Any]],
    missing_files: list[str],
    notes: list[str],
    malformed: bool,
) -> dict[str, Any]:
    metric_names = {str(row["metric_name"]) for row in long_rows}
    gradient_values = [row for row in long_rows if row["metric_type"] == "gradient_norm"]
    device_values = [row for row in long_rows if row["metric_type"] == "device_stat"]
    available_for_optional = set(metric_names)
    if gradient_values:
        available_for_optional.update({"gradient_norm_mean", "gradient_norm_std"})
    if device_values:
        available_for_optional.add("device_stat")

    missing_required = [name for name in REQUIRED_METRICS if name not in metric_names]
    missing_optional = [name for name in OPTIONAL_METRICS if name not in available_for_optional]

    epoch_col = pick_epoch_column(metrics_df)
    n_epochs = 0
    first_epoch = np.nan
    last_epoch = np.nan
    duplicate_epochs = False
    if epoch_col is not None and not metrics_df.empty:
        epochs = pd.to_numeric(metrics_df[epoch_col], errors="coerce").dropna()
        if not epochs.empty:
            n_epochs = int(epochs.nunique())
            first_epoch = int(epochs.min())
            last_epoch = int(epochs.max())
            duplicate_epochs = bool(epochs.duplicated().any())

    has_nan = False
    if not metrics_df.empty:
        numeric = metrics_df.apply(pd.to_numeric, errors="coerce")
        metric_cols = [col for col in metrics_df.columns if is_metric_column(col)]
        has_nan = bool(numeric[metric_cols].isna().any().any()) if metric_cols else False

    accuracy_values = [
        row["metric_value"]
        for row in long_rows
        if row["metric_type"] == "accuracy" and row.get("metric_value") is not None
    ]
    if accuracy_values:
        max_acc = max(accuracy_values)
        min_acc = min(accuracy_values)
        if max_acc <= 1.5:
            notes.append("accuracy scale detected: 0-1")
        elif min_acc >= 0 and max_acc <= 100.5:
            notes.append("accuracy scale detected: 0-100")
        else:
            notes.append("accuracy scale detected: mixed or unusual")

    warning_notes = [
        note for note in notes if not str(note).startswith("accuracy scale detected:")
    ]

    if malformed:
        status = "failed"
    elif missing_required or missing_files or warning_notes:
        status = "warning"
    else:
        status = "ok"

    return {
        "model_config_name": identity.model_config_name,
        "seed": seed,
        "status": status,
        "missing_files": "; ".join(missing_files),
        "missing_required_metrics": "; ".join(missing_required),
        "missing_optional_metrics": "; ".join(missing_optional),
        "n_epochs_found": n_epochs,
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "has_nan": has_nan,
        "duplicate_epochs": duplicate_epochs,
        "notes": "; ".join(dict.fromkeys(notes)),
    }


def missing_expected_files(model_dir: Path, seed_dir: Path | None, metrics_source: Path | None) -> list[str]:
    missing = []
    if not (model_dir / ".model").exists():
        missing.append(".model")
    if not (model_dir / ".hydra" / "config.yaml").exists() and not (
        seed_dir is not None and (seed_dir / ".hydra" / "config.yaml").exists()
    ):
        missing.append(".hydra/config.yaml")
    if not (model_dir / ".hydra" / "overrides.yaml").exists() and not (
        seed_dir is not None and (seed_dir / ".hydra" / "overrides.yaml").exists()
    ):
        missing.append(".hydra/overrides.yaml")
    if not (model_dir / "model.info").exists() and not (seed_dir is not None and (seed_dir / "model.info").exists()):
        missing.append("model.info")
    if metrics_source is None:
        missing.append("<seed>/metrics/metrics.csv")
    return missing


def extract_seed(
    models_root: Path,
    model_dir: Path,
    seed_dir: Path | None,
    include_tensorboard_events: bool,
) -> tuple[SeedExtraction, dict[str, Any] | None]:
    seed = int(seed_dir.name) if seed_dir is not None and seed_dir.name.isdigit() else None
    cfg, config_source = load_hydra_config(model_dir, seed_dir)
    overrides, _ = load_overrides(model_dir, seed_dir)
    hparams, _ = load_hparams(seed_dir)
    identity = build_identity(models_root, model_dir, cfg, hparams)
    result = SeedExtraction()

    try:
        metrics_df, metrics_source, metric_notes, malformed = read_metrics(seed_dir)
        result.notes.extend(metric_notes)
        result.malformed = malformed

        long_rows, raw_metric_map = metrics_to_long_rows(metrics_df, metrics_source, identity, seed)
        result.long_rows.extend(long_rows)
        result.raw_metric_map.update(raw_metric_map)
        result.metric_names.update(row["metric_name"] for row in long_rows)

        if include_tensorboard_events:
            tb_rows, tb_notes, tb_map = parse_tensorboard_scalars(seed_dir, identity, seed)
            result.long_rows.extend(tb_rows)
            result.notes.extend(tb_notes)
            result.raw_metric_map.update(tb_map)
            result.metric_names.update(row["metric_name"] for row in tb_rows)

        static_row, info, info_source, static_notes = build_static_row(
            model_dir=model_dir,
            seed_dir=seed_dir,
            identity=identity,
            cfg=cfg,
            hparams=hparams,
            overrides=overrides,
            metrics_df=metrics_df,
        )
        result.notes.extend(static_notes)
        result.static_row = static_row

        missing_files = missing_expected_files(model_dir, seed_dir, metrics_source)
        result.quality_row = quality_from_metrics(
            identity=identity,
            seed=seed,
            metrics_df=metrics_df,
            long_rows=result.long_rows,
            missing_files=missing_files,
            notes=list(result.notes),
            malformed=result.malformed,
        )
        quantum_row = build_quantum_row(identity, cfg, hparams, static_row, info_source, config_source)
        return result, quantum_row
    except Exception as exc:
        log.exception("Failed to extract %s seed=%s", identity.model_config_name, seed)
        result.malformed = True
        result.quality_row = {
            "model_config_name": identity.model_config_name,
            "seed": seed,
            "status": "failed",
            "missing_files": "; ".join(missing_expected_files(model_dir, seed_dir, None)),
            "missing_required_metrics": "; ".join(REQUIRED_METRICS),
            "missing_optional_metrics": "; ".join(OPTIONAL_METRICS),
            "n_epochs_found": 0,
            "first_epoch": np.nan,
            "last_epoch": np.nan,
            "has_nan": False,
            "duplicate_epochs": False,
            "notes": str(exc),
        }
        return result, None


def build_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=WIDE_BASE_COLUMNS)

    work = long_df.copy()
    work = work[~pd.isna(work["epoch"])].copy()
    if work.empty:
        return pd.DataFrame(columns=WIDE_BASE_COLUMNS)
    work["epoch"] = pd.to_numeric(work["epoch"], errors="coerce")
    work["step_sort"] = pd.to_numeric(work["step"], errors="coerce")
    work["step_sort"] = work["step_sort"].fillna(-1)

    index_cols = [
        "model_config_name",
        "canonical_model_name",
        "family",
        "architecture",
        "variant_configuration",
        "seed",
        "epoch",
    ]
    work = work.sort_values(index_cols + ["metric_name", "step_sort"], kind="mergesort")
    last_values = (
        work.groupby(index_cols + ["metric_name"], dropna=False, as_index=False)
        .tail(1)
        .pivot_table(
            index=index_cols,
            columns="metric_name",
            values="metric_value",
            aggfunc="last",
        )
        .reset_index()
    )
    last_values.columns.name = None

    grad = work[work["metric_type"] == "gradient_norm"]
    if not grad.empty:
        grad_summary = (
            grad.groupby(index_cols, dropna=False)["metric_value"]
            .agg(gradient_norm_mean="mean", gradient_norm_std="std")
            .reset_index()
        )
        last_values = last_values.merge(grad_summary, on=index_cols, how="left", suffixes=("", "_computed"))
        for col in ["gradient_norm_mean", "gradient_norm_std"]:
            computed = f"{col}_computed"
            if computed in last_values.columns:
                if col in last_values.columns:
                    last_values[col] = last_values[col].combine_first(last_values[computed])
                    last_values = last_values.drop(columns=[computed])
                else:
                    last_values = last_values.rename(columns={computed: col})

    for column in WIDE_BASE_COLUMNS:
        if column not in last_values.columns:
            last_values[column] = np.nan

    extra_cols = [col for col in last_values.columns if col not in WIDE_BASE_COLUMNS]
    return last_values[WIDE_BASE_COLUMNS + sorted(extra_cols)]


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    extra = [column for column in out.columns if column not in columns]
    return out[columns + extra]


def write_summary(
    output_dir: Path,
    models_root: Path,
    model_dirs: list[Path],
    quality_df: pd.DataFrame,
    raw_metric_counter: Counter[str],
    metric_variant_map: dict[str, set[str]],
) -> None:
    valid_anchor_count = len(model_dirs)
    seeds_found = int(quality_df["seed"].notna().sum()) if not quality_df.empty else 0
    metrics_parsed = int((~quality_df["missing_files"].str.contains("<seed>/metrics/metrics.csv", na=False)).sum()) if not quality_df.empty else 0
    missing_metrics = quality_df[
        quality_df["missing_files"].str.contains("<seed>/metrics/metrics.csv", na=False)
    ] if not quality_df.empty else pd.DataFrame()
    malformed = quality_df[quality_df["status"].eq("failed")] if not quality_df.empty else pd.DataFrame()

    variants_lines = []
    for normalized, raw_names in sorted(metric_variant_map.items()):
        variants = ", ".join(sorted(raw_names))
        variants_lines.append(f"- `{normalized}`: {variants}")
    if not variants_lines:
        variants_lines.append("- No scalar metric columns detected.")

    missing_lines = []
    for _, row in missing_metrics.head(30).iterrows():
        missing_lines.append(f"- `{row['model_config_name']}` seed `{row['seed']}`")
    if len(missing_metrics) > 30:
        missing_lines.append(f"- ... {len(missing_metrics) - 30} more")
    if not missing_lines:
        missing_lines.append("- None")

    malformed_lines = []
    for _, row in malformed.head(30).iterrows():
        malformed_lines.append(f"- `{row['model_config_name']}` seed `{row['seed']}`: {row['notes']}")
    if len(malformed) > 30:
        malformed_lines.append(f"- ... {len(malformed) - 30} more")
    if not malformed_lines:
        malformed_lines.append("- None")

    text = f"""# Training Dynamics Extraction Summary

Generated at: {datetime.now().isoformat(timespec="seconds")}

Input models root: `{models_root}`
Output directory: `{output_dir}`

## Counts

- Model folders found: {len(model_dirs)}
- Valid `.model` anchors: {valid_anchor_count}
- Seeds found: {seeds_found}
- Metrics files parsed: {metrics_parsed}

## Models With Missing Metrics

{chr(10).join(missing_lines)}

## Models With Malformed Metrics

{chr(10).join(malformed_lines)}

## Detected Metric Naming Variants

{chr(10).join(variants_lines)}

## Assumptions Made

- Directories containing `.model` are valid model configuration folders.
- Numeric child directories are seed folders.
- `metrics.csv` is preferred, then `metrics.json`, then `logger.pkl` if it can be read without model imports.
- Accuracy values are preserved as stored. Scale notes are written to `extraction_quality_report.csv`.
- Duplicate epoch rows are kept in the long table with their step values. The wide table keeps the last value by step for each epoch and metric.
- Parameter counts, training time, and epoch counts are parsed from `model.info` when present; otherwise the script uses lightweight metadata fallbacks.
- Checkpoints are counted by filename only. Checkpoint tensor contents are not loaded.

## Recommended Next Steps

- Review `extraction_quality_report.csv` before plotting.
- Check warning rows with missing required metrics or unusual accuracy scale notes.
- Use `training_curves_long.csv` for flexible plotting and `training_curves_wide.csv` for epoch-aligned summaries.
- For quantum resource claims, treat blank fields in `quantum_static_metadata.csv` as unavailable from artifacts rather than zero.
"""
    (output_dir / "extraction_summary.md").write_text(text, encoding="utf-8")


def check_overwrite(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        files = "\n".join(f"- {path}" for path in existing)
        raise SystemExit(
            "Output files already exist. Re-run with --overwrite true or choose a new --output-dir.\n"
            f"{files}"
        )


def export_training_dynamics(
    models_root: Path,
    output_dir: Path,
    include_tensorboard_events: bool,
    overwrite: bool,
) -> None:
    models_root = models_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not models_root.is_dir():
        raise FileNotFoundError(f"models root not found: {models_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    check_overwrite(output_dir, overwrite)

    model_dirs = find_model_dirs(models_root)
    log.info("Found %s .model anchors under %s", len(model_dirs), models_root)

    long_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    quantum_by_model: dict[str, dict[str, Any]] = {}
    raw_metric_counter: Counter[str] = Counter()
    metric_variant_map: dict[str, set[str]] = defaultdict(set)

    for model_dir in tqdm(model_dirs, desc="Model folders"):
        seed_dirs = iter_seed_dirs(model_dir)
        if not seed_dirs:
            seed_dirs = [None]  # type: ignore[list-item]

        for seed_dir in seed_dirs:
            result, quantum_row = extract_seed(
                models_root=models_root,
                model_dir=model_dir,
                seed_dir=seed_dir,
                include_tensorboard_events=include_tensorboard_events,
            )
            long_rows.extend(result.long_rows)
            if result.static_row:
                static_rows.append(result.static_row)
            if result.quality_row:
                quality_rows.append(result.quality_row)
            for raw_name, normalized in result.raw_metric_map.items():
                raw_metric_counter[raw_name] += 1
                metric_variant_map[normalized].add(raw_name)
            if quantum_row is not None and quantum_row["model_config_name"] not in quantum_by_model:
                quantum_by_model[str(quantum_row["model_config_name"])] = quantum_row

    long_df = ensure_columns(pd.DataFrame(long_rows), LONG_COLUMNS)
    wide_df = build_wide(long_df)
    static_df = ensure_columns(pd.DataFrame(static_rows), STATIC_COLUMNS)
    quantum_df = ensure_columns(pd.DataFrame(quantum_by_model.values()), QUANTUM_COLUMNS)
    quality_df = ensure_columns(pd.DataFrame(quality_rows), QUALITY_COLUMNS)

    long_df.to_csv(output_dir / "training_curves_long.csv", index=False)
    wide_df.to_csv(output_dir / "training_curves_wide.csv", index=False)
    static_df.to_csv(output_dir / "model_static_metadata.csv", index=False)
    quantum_df.to_csv(output_dir / "quantum_static_metadata.csv", index=False)
    quality_df.to_csv(output_dir / "extraction_quality_report.csv", index=False)
    write_summary(output_dir, models_root, model_dirs, quality_df, raw_metric_counter, metric_variant_map)

    log.info("Wrote %s rows to %s", len(long_df), output_dir / "training_curves_long.csv")
    log.info("Wrote %s rows to %s", len(wide_df), output_dir / "training_curves_wide.csv")
    log.info("Wrote %s rows to %s", len(static_df), output_dir / "model_static_metadata.csv")
    log.info("Wrote %s rows to %s", len(quantum_df), output_dir / "quantum_static_metadata.csv")
    log.info("Wrote %s rows to %s", len(quality_df), output_dir / "extraction_quality_report.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export training dynamics from saved model artifacts without loading checkpoints."
    )
    parser.add_argument("--models-root", required=True, type=Path, help="Root to crawl for .model anchors.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for exported CSV/Markdown files.")
    parser.add_argument(
        "--include-tensorboard-events",
        default=False,
        type=parse_bool,
        help="Parse TensorBoard scalar event files if tensorboard is installed. Use true or false.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        type=parse_bool,
        help="Overwrite existing output files in --output-dir. Use true or false.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_parser().parse_args()
    export_training_dynamics(
        models_root=args.models_root,
        output_dir=args.output_dir,
        include_tensorboard_events=args.include_tensorboard_events,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
