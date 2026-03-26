from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from loading import crawl_experiments, load_model

log = logging.getLogger(__name__)


# ============================================================
# Data containers
# ============================================================

@dataclass
class SeedContext:
    """All objects needed to extract one output row."""

    root_dir: Path
    model_dir: Path
    relative_model_dir: str
    seed: int
    load_mode: str
    checkpoint_path: Optional[Path]
    model: Any
    hydra_config: Dict[str, Any]
    metrics_df: pd.DataFrame
    model_info_text: str


# ============================================================
# Generic helpers
# ============================================================


def safe_get(obj: Any, *path: str, default: Any = None) -> Any:
    """Best-effort nested lookup for dict-like or attribute-based objects."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur



def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")



def read_hydra_config(model_dir: Path) -> Dict[str, Any]:
    hydra_cfg = model_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return {}
    return OmegaConf.to_container(OmegaConf.load(str(hydra_cfg)), resolve=True)  # type: ignore[return-value]



def read_metrics_df(seed_dir: Path) -> pd.DataFrame:
    csv_path = seed_dir / "metrics" / "metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        log.warning("Failed to read metrics CSV at %s: %s", csv_path, exc)
        return pd.DataFrame()



def parse_model_info(seed_dir: Path) -> str:
    return read_text_if_exists(seed_dir / "model.info")

def _parse_number_token(token: str) -> Optional[int]:
    token = token.strip().replace(",", "")
    if not token or token == "--":
        return None
    if not re.fullmatch(r"\d+", token):
        return None
    return int(token)


def parse_model_info_summary(text: str) -> Dict[str, Optional[float]]:
    """Parse the persisted torchinfo-style model.info summary.

    Handles both:
    - Total training time: X seconds
    - Total training time (per seed): X seconds
    """
    out: Dict[str, Optional[float]] = {
        "total_params": None,
        "trainable_params": None,
        "non_trainable_params": None,
        "trained_epochs": None,
        "training_time_seconds": None,
        "training_time_per_epoch_seconds": None,
        "classical_params_from_lines": None,
        "quantum_params_from_lines": None,
    }

    patterns = {
        "total_params": r"Total params:\s*([\d,]+)",
        "trainable_params": r"Trainable params:\s*([\d,]+)",
        "non_trainable_params": r"Non-trainable params:\s*([\d,]+)",
        "trained_epochs": r"Trained epochs:\s*(\d+)",
        "training_time_per_epoch_seconds": r"Training time per epoch:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
        "training_time_seconds": r"Total training time(?:\s*\(per seed\))?:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if not m:
            continue
        value = m.group(1).replace(",", "")
        out[key] = float(value)

    # Split params by parsing per-layer rows from torchinfo summary
    classical_total = 0
    quantum_total = 0
    saw_any_param_line = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or "Param #" in line or "===" in line:
            continue

        m = re.match(r"^(?P<layer>.+?)\s{2,}(?P<params>[\d,]+|--)\s{2,}.*$", line)
        if not m:
            continue

        param_count = _parse_number_token(m.group("params"))
        if param_count is None:
            continue

        saw_any_param_line = True
        layer_name = m.group("layer").lower()

        is_quantum = any(token in layer_name for token in [
            "qlayer",
            "torchlayer",
            "vqc",
            "vql",
            "quantum",
            "qnode",
            "ansatz",
            "quanv",
        ])

        if is_quantum:
            quantum_total += param_count
        else:
            classical_total += param_count

    if saw_any_param_line:
        out["classical_params_from_lines"] = float(classical_total)
        out["quantum_params_from_lines"] = float(quantum_total)

    return out


def parse_training_time_from_model_info(text: str) -> Optional[float]:
    return parse_model_info_summary(text)["training_time_seconds"]


def parse_time_per_epoch_from_model_info(text: str) -> Optional[float]:
    return parse_model_info_summary(text)["training_time_per_epoch_seconds"]


def parse_trained_epochs_from_model_info(text: str) -> Optional[int]:
    value = parse_model_info_summary(text)["trained_epochs"]
    return int(value) if value is not None else None

def infer_checkpoint_path(seed_dir: Path, load_mode: str) -> Optional[Path]:
    final_ckpt = seed_dir / "final.ckpt"
    if load_mode == "final":
        return final_ckpt if final_ckpt.exists() else None

    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        return final_ckpt if final_ckpt.exists() else None

    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return final_ckpt if final_ckpt.exists() else None

    def score(path: Path) -> float:
        m = re.search(r"loss=(\d+\.\d+)", path.name)
        return float(m.group(1)) if m else float("inf")

    ckpts = sorted(ckpts, key=score)
    return ckpts[0]



def find_family_in_path(parts: Iterable[str]) -> str:
    families = {"classical": "Classical", "hybrid": "Hybrid", "quantum": "Quantum"}
    for part in parts:
        key = part.lower()
        if key in families:
            return families[key]
    return "Unknown"



def normalize_architecture_name(name: str) -> str:
    mapping = {
        "convnet": "ConvNet",
        "lenet": "LeNet",
        "mlp": "MLP",
        "mlpmixer": "MLPMixer",
        "resnet18": "ResNet18",
        "hvql": "HVQL",
        "hvqm": "HVQM",
        "hpvqm": "HPVQM",
        "hpvql": "HPVQL",
        "hquanv": "HQuanv",
        "hqe": "HQE",
        "drophvql": "DropHVQL",
        "reshvql": "ResHVQL",
        "vqm50": "VQM50",
        "vqm100": "VQM100",
        "vqm200": "VQM200",
        "vqm50-basic": "VQM50",
        "vqm50-random": "VQM50",
        "vqm50-strong": "VQM50",
        "vqm50-identity": "VQM50",
        "vqm50-noentangling": "VQM50",
    }
    return mapping.get(name.lower(), name)



def infer_architecture_and_knobs(relative_model_dir: str) -> tuple[str, str, str]:
    """
    Infer group/family/architecture/knobs from the experiment directory structure.
    This mirrors the uploaded experiment tree and keeps parsing logic isolated.
    """
    parts = Path(relative_model_dir).parts
    family = find_family_in_path(parts)

    architecture = "Unknown"
    knobs = ""

    if len(parts) >= 1:
        architecture = normalize_architecture_name(parts[-1])

    # Handle layouts where the leaf folder is the knob directory.
    if "=" in parts[-1] or parts[-1].lower() in {"dropout", "residuals"}:
        if len(parts) >= 2:
            architecture = normalize_architecture_name(parts[-2])
            knobs = parts[-1]
    else:
        # VQM50-Basic style folders under Quantum
        leaf = parts[-1]
        m = re.match(r"(?P<arch>VQM\d+)-(?P<ansatz>.+)", leaf, flags=re.IGNORECASE)
        if m:
            architecture = normalize_architecture_name(m.group("arch"))
            knobs = f"ansatz={m.group('ansatz').lower()}"

    if not knobs:
        lower_leaf = parts[-1].lower()
        if lower_leaf == "dropout":
            knobs = "dropout enabled"
        elif lower_leaf == "residuals":
            knobs = "residual connections enabled"
        elif architecture.lower() in {"lenet", "mlpmixer", "resnet18", "hqe", "hquanv", "hpvql", "drophvql", "reshvql", "vqm100", "vqm200"}:
            knobs = "baseline configuration"

    return family, architecture, knobs



def _last_valid(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])



def metric_last(metrics_df: pd.DataFrame, *candidates: str) -> Optional[float]:
    for col in candidates:
        if col in metrics_df.columns:
            return _last_valid(metrics_df[col])
    return None



def metric_min(metrics_df: pd.DataFrame, *candidates: str) -> Optional[float]:
    for col in candidates:
        if col in metrics_df.columns:
            valid = metrics_df[col].dropna()
            if not valid.empty:
                return float(valid.min())
    return None



def metric_max(metrics_df: pd.DataFrame, *candidates: str) -> Optional[float]:
    for col in candidates:
        if col in metrics_df.columns:
            valid = metrics_df[col].dropna()
            if not valid.empty:
                return float(valid.max())
    return None



def metric_epoch_of_min(metrics_df: pd.DataFrame, metric_name: str) -> Optional[int]:
    if metric_name not in metrics_df.columns or "epoch" not in metrics_df.columns:
        return None
    series = metrics_df[metric_name]
    if series.dropna().empty:
        return None
    idx = series.idxmin()
    value = metrics_df.loc[idx, "epoch"]
    return int(value) if pd.notna(value) else None

def named_parameters_list(model: Any) -> List[tuple[str, torch.nn.Parameter]]:
    try:
        return list(model.named_parameters())
    except Exception:
        return []



def _is_quantum_parameter_name(name: str) -> bool:
    name = name.lower()
    patterns = [
        "quantum",
        "qnode",
        "qlayer",
        "circuit",
        "ansatz",
        "rotation",
        "theta",
        "phi",
        "lambda",
        "weights_q",
        "qweights",
        "q_params",
        "quantum_params",
    ]
    return any(token in name for token in patterns)


# ============================================================
# Column extractors
# One refactored function per column.
# ============================================================


def extract_family(ctx: SeedContext) -> str:
    family, _, _ = infer_architecture_and_knobs(ctx.relative_model_dir)
    return family



def extract_architecture(ctx: SeedContext) -> str:
    _, architecture, _ = infer_architecture_and_knobs(ctx.relative_model_dir)
    return architecture



def extract_variant_configuration_knobs(ctx: SeedContext) -> str:
    _, architecture, inferred_knobs = infer_architecture_and_knobs(ctx.relative_model_dir)
    model_cfg = safe_get(ctx.hydra_config, "model", default={}) or {}

    if inferred_knobs and inferred_knobs != "baseline configuration":
        return inferred_knobs

    candidates = []
    for key in [
        "hidden_sizes",
        "conv_channels",
        "fc_sizes",
        "ansatz",
        "qdepth",
        "qblocks",
        "dropout",
        "dropout_p",
        "residual",
        "residuals",
        "n_qubits",
        "qubits",
    ]:
        if key in model_cfg:
            value = model_cfg[key]
            candidates.append(f"{key}={value}")

    if candidates:
        return ", ".join(candidates)

    if architecture in {"LeNet", "MLPMixer", "ResNet18", "HQE", "HQuanv", "HPVQL", "DropHVQL", "ResHVQL", "VQM100", "VQM200"}:
        return "baseline configuration"

    return inferred_knobs or ""



def extract_params_total(ctx: SeedContext) -> int:
    return int(sum(p.numel() for p in ctx.model.parameters()))



def extract_params_quantum(ctx: SeedContext) -> int:
    total = 0
    for name, param in named_parameters_list(ctx.model):
        if _is_quantum_parameter_name(name):
            total += param.numel()
    return int(total)



def extract_params_classical(ctx: SeedContext) -> int:
    total = extract_params_total(ctx)
    q_total = extract_params_quantum(ctx)
    return int(max(total - q_total, 0))



def extract_qubits(ctx: SeedContext) -> Optional[int]:
    """Extract the effective number of qubits from the loaded object, not from Hydra.

    In this codebase, the qubit count is determined by the instantiated quantum
    embedding / ansatz stack. For example, BaseAnsatz stores ``n_qubits`` on the
    object, VQC exposes ``ansatz.n_qubits`` through ``self.ansatz``, and VQM
    owns a ``self.vqc`` module built on top of VQC. That means the loaded model
    object is the reliable source of truth for qubit count.
    """
    candidates = [
        getattr(ctx.model, "n_qubits", None),
        getattr(getattr(ctx.model, "vqc", None), "output_size", None),
        getattr(getattr(getattr(ctx.model, "vqc", None), "ansatz", None), "n_qubits", None),
        getattr(getattr(ctx.model, "qlayer", None), "output_size", None),
        getattr(getattr(getattr(ctx.model, "qlayer", None), "vqc", None), "output_size", None),
        getattr(getattr(getattr(getattr(ctx.model, "qlayer", None), "vqc", None), "ansatz", None), "n_qubits", None),
    ]
    for value in candidates:
        if isinstance(value, (int, float)):
            return int(value)
    return None



def extract_q_depth(ctx: SeedContext) -> Optional[int]:
    """Extract circuit depth from the loaded object whenever possible.

    The quantum stack exposes an object-level depth computation through
    ``BaseAnsatz.circuit_depth()``. This is preferable to config parsing because
    the real instantiated object may differ from the raw Hydra structure.
    """
    # 1) Exact object-level circuit depth if available
    ansatz_candidates = [
        getattr(getattr(ctx.model, "vqc", None), "ansatz", None),
        getattr(getattr(getattr(ctx.model, "qlayer", None), "vqc", None), "ansatz", None),
        getattr(getattr(ctx.model, "ansatz", None), "circuit_depth", None),
    ]

    for candidate in ansatz_candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "circuit_depth") and callable(candidate.circuit_depth):
            try:
                depth = candidate.circuit_depth()
                if depth is not None:
                    return int(depth)
            except Exception:
                pass

    # 2) Fall back to the configured n_qdepth if that is all we have
    model_hparams = getattr(ctx.model, "hparams", {})
    for key in ["n_qdepth", "qdepth", "n_layers", "depth"]:
        value = safe_get(model_hparams, key)
        if value is not None and isinstance(value, (int, float)):
            return int(value)

    model_cfg = safe_get(ctx.hydra_config, "model", default={}) or {}
    for key in ["n_qdepth", "qdepth", "q_depth", "circuit_depth", "depth"]:
        value = model_cfg.get(key)
        if value is not None and isinstance(value, (int, float)):
            return int(value)
    return None



def _iter_quantum_parameter_tensors(model: Any) -> Iterable[torch.nn.Parameter]:
    """Yield parameter tensors belonging to known quantum submodules.

    In this repository, quantum weights live inside PennyLane TorchLayer objects,
    typically reachable through:
    - model.vqc.qlayer               (pure VQM / VQL-style)
    - model.qlayer                   (hybrid layer object)
    - model.qlayer.vqc.qlayer        (HVQL / HPVQL-style)
    """
    seen = set()
    submodules = [
        getattr(getattr(ctx_model := model, "vqc", None), "qlayer", None),
        getattr(model, "qlayer", None),
        getattr(getattr(getattr(model, "qlayer", None), "vqc", None), "qlayer", None),
    ]
    for submodule in submodules:
        if submodule is None:
            continue
        try:
            for param in submodule.parameters():
                pid = id(param)
                if pid not in seen:
                    seen.add(pid)
                    yield param
        except Exception:
            continue

def extract_params_total(ctx: SeedContext) -> int:
    parsed = parse_model_info_summary(ctx.model_info_text)
    total = parsed.get("total_params")
    if total is not None:
        return int(total)
    return int(sum(p.numel() for p in ctx.model.parameters()))


def extract_params_quantum(ctx: SeedContext) -> int:
    parsed = parse_model_info_summary(ctx.model_info_text)
    q_from_lines = parsed.get("quantum_params_from_lines")
    if q_from_lines is not None:
        return int(q_from_lines)

    quantum_params = list(_iter_quantum_parameter_tensors(ctx.model))
    if quantum_params:
        return int(sum(p.numel() for p in quantum_params))

    total = 0
    for name, param in named_parameters_list(ctx.model):
        if _is_quantum_parameter_name(name):
            total += param.numel()
    return int(total)


def extract_params_classical(ctx: SeedContext) -> int:
    parsed = parse_model_info_summary(ctx.model_info_text)
    c_from_lines = parsed.get("classical_params_from_lines")
    if c_from_lines is not None:
        return int(c_from_lines)

    total = extract_params_total(ctx)
    q_total = extract_params_quantum(ctx)
    return int(max(total - q_total, 0))


def extract_epochs(ctx: SeedContext) -> Optional[int]:
    parsed = parse_model_info_summary(parse_model_info(ctx.model_dir))
    epochs = parsed.get("trained_epochs")
    if epochs is not None:
        return int(epochs)

    model_cfg_epochs = safe_get(ctx.hydra_config, "training", "epochs")
    if model_cfg_epochs is not None:
        return int(model_cfg_epochs)
    return None


def extract_training_time_seconds(ctx: SeedContext) -> Optional[float]:
    performance_path = ctx.model_dir / "performance.json"
    if performance_path.exists():
        try:
            with performance_path.open("r", encoding="utf-8") as f:
                performance = json.load(f)

            seed_times = performance.get("seed_times", {})
            seed_key = str(ctx.seed)
            if seed_key in seed_times:
                return float(seed_times[seed_key])

            average = performance.get("average")
            if average is not None:
                return float(average)
        except Exception as exc:
            log.warning("Failed to read performance.json at %s: %s", performance_path, exc)

    return parse_training_time_from_model_info(parse_model_info(ctx.model_dir))

def extract_param_ratio_q_classical(ctx: SeedContext) -> Optional[float]:
    classical = extract_params_classical(ctx)
    quantum = extract_params_quantum(ctx)
    if classical == 0:
        return None
    return float(quantum / classical)

def extract_batch_size(ctx: SeedContext) -> Optional[int]:
    for key in ["train_batch_size", "batch_size"]:
        value = safe_get(ctx.hydra_config, "dataset", key)
        if value is not None:
            return int(value)
    return None



def extract_learning_rate(ctx: SeedContext) -> Optional[float]:
    lr = safe_get(ctx.hydra_config, "training", "optimiser", "learning_rate")
    if lr is not None:
        return float(lr)
    lr = safe_get(ctx.hydra_config, "training", "learning_rate")
    if lr is not None:
        return float(lr)
    return safe_get(ctx.model, "hparams", "learning_rate")



def extract_optimizer(ctx: SeedContext) -> Optional[str]:
    value = safe_get(ctx.hydra_config, "training", "optimiser", "name")
    if value is not None:
        return str(value)
    return safe_get(ctx.model, "hparams", "optimizer")



def extract_device(ctx: SeedContext) -> str:
    accelerator = safe_get(ctx.hydra_config, "device", "accelerator")
    devices = safe_get(ctx.hydra_config, "device", "devices")
    if accelerator is None and devices is None:
        return ""
    return f"accelerator={accelerator}, devices={devices}"



def extract_seed(ctx: SeedContext) -> int:
    return int(ctx.seed)



def extract_num_seeds_aggregated(ctx: SeedContext) -> int:
    # Per-row extraction is one seed at a time.
    return 1



def extract_checkpoint_type(ctx: SeedContext) -> str:
    return "best checkpoint" if ctx.load_mode == "best" else "final snapshot"



def extract_selection_metric(ctx: SeedContext) -> str:
    if ctx.load_mode == "best":
        return "val/loss"
    return "final epoch"



def extract_selection_value(ctx: SeedContext) -> Optional[float]:
    if ctx.load_mode == "best":
        return metric_min(ctx.metrics_df, "val_loss", "val/loss")
    return None



def extract_epoch_selected(ctx: SeedContext) -> Optional[int]:
    if ctx.load_mode == "best":
        return metric_epoch_of_min(ctx.metrics_df, "val_loss")
    if not ctx.metrics_df.empty and "epoch" in ctx.metrics_df.columns:
        valid = ctx.metrics_df["epoch"].dropna()
        if not valid.empty:
            return int(valid.iloc[-1])
    return None



def extract_train_accuracy(ctx: SeedContext) -> Optional[float]:
    return metric_last(ctx.metrics_df, "train_acc", "train/acc")



def extract_validation_accuracy(ctx: SeedContext) -> Optional[float]:
    return metric_last(ctx.metrics_df, "val_acc", "val/acc")



def extract_test_accuracy(ctx: SeedContext) -> Optional[float]:
    # TODO: test metrics are not ready in the current pipeline.
    return None



def extract_train_loss(ctx: SeedContext) -> Optional[float]:
    return metric_last(ctx.metrics_df, "train_loss", "train/loss")



def extract_validation_loss(ctx: SeedContext) -> Optional[float]:
    return metric_last(ctx.metrics_df, "val_loss", "val/loss")



def extract_test_loss(ctx: SeedContext) -> Optional[float]:
    # TODO: test metrics are not ready in the current pipeline.
    return None



def extract_generalization_gap_acc(ctx: SeedContext) -> Optional[float]:
    train_acc = extract_train_accuracy(ctx)
    val_acc = extract_validation_accuracy(ctx)
    if train_acc is None or val_acc is None:
        return None
    return float(train_acc - val_acc)



def extract_generalization_gap_loss(ctx: SeedContext) -> Optional[float]:
    train_loss = extract_train_loss(ctx)
    val_loss = extract_validation_loss(ctx)
    if train_loss is None or val_loss is None:
        return None
    return float(val_loss - train_loss)



def extract_accuracy_std_seeds(ctx: SeedContext) -> Optional[float]:
    # TODO: this is a per-seed export. Seed aggregation should be computed in a second pass.
    return None



def extract_best_worst_acc_gap(ctx: SeedContext) -> Optional[float]:
    # TODO: this is a per-seed export. Seed aggregation should be computed in a second pass.
    return None



def extract_convergence_epoch(ctx: SeedContext) -> Optional[int]:
    # Best-effort approximation: first epoch reaching 99% of the best validation accuracy.
    if "epoch" not in ctx.metrics_df.columns or "val_acc" not in ctx.metrics_df.columns:
        return None
    df = ctx.metrics_df[["epoch", "val_acc"]].dropna()
    if df.empty:
        return None
    best = float(df["val_acc"].max())
    threshold = 0.99 * best
    reached = df[df["val_acc"] >= threshold]
    if reached.empty:
        return None
    return int(reached.iloc[0]["epoch"])


def _flatten_grad_l2_values(ctx: SeedContext) -> list[float]:
    metrics = getattr(ctx.model, "metrics", None)
    if metrics is None:
        return []

    grad_l2 = metrics.get("grad_l2")
    if not isinstance(grad_l2, dict):
        return []

    values: list[float] = []
    for _, history in grad_l2.items():
        if not isinstance(history, list):
            continue
        for v in history:
            if v is None:
                continue
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue
    return values


def _grad_l2_histories(ctx: SeedContext) -> dict[str, list[float]]:
    metrics = getattr(ctx.model, "metrics", None)
    if metrics is None:
        return {}

    grad_l2 = metrics["grad_l2"]
    if not isinstance(grad_l2, dict):
        return {}

    cleaned: dict[str, list[float]] = {}
    for layer_name, history in grad_l2.items():
        if not isinstance(history, list):
            continue

        vals = []
        for v in history:
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue

        if vals:
            cleaned[layer_name] = vals

    return cleaned

def extract_gradient_norm_mean(ctx: SeedContext) -> Optional[float]:
    histories = _grad_l2_histories(ctx)
    if not histories:
        return None

    n_layers = len(histories)
    n_points = sum(len(v) for v in histories.values())

    # reject broken one-shot logging
    if n_points < 2 * n_layers:
        return None

    flat = [x for values in histories.values() for x in values]
    if not flat:
        return None

    return float(pd.Series(flat, dtype=float).mean())


def extract_gradient_norm_std(ctx: SeedContext) -> Optional[float]:
    histories = _grad_l2_histories(ctx)
    if not histories:
        return None

    n_layers = len(histories)
    n_points = sum(len(v) for v in histories.values())

    if n_points < 2 * n_layers:
        return None

    flat = [x for values in histories.values() for x in values]
    if len(flat) < 2:
        return None

    return float(pd.Series(flat, dtype=float).std(ddof=1))


def extract_gradient_norm_final(ctx: SeedContext) -> Optional[float]:
    histories = _grad_l2_histories(ctx)
    if not histories:
        return None

    n_layers = len(histories)
    n_points = sum(len(v) for v in histories.values())

    if n_points < 2 * n_layers:
        return None

    finals = [values[-1] for values in histories.values() if values]
    if not finals:
        return None

    return float(pd.Series(finals, dtype=float).mean())


def extract_gradient_norm_max(ctx: SeedContext) -> Optional[float]:
    histories = _grad_l2_histories(ctx)
    if not histories:
        return None

    n_layers = len(histories)
    n_points = sum(len(v) for v in histories.values())

    if n_points < 2 * n_layers:
        return None

    flat = [x for values in histories.values() for x in values]
    if not flat:
        return None

    return float(max(flat))


def extract_clean_accuracy_attack_baseline(ctx: SeedContext) -> Optional[float]:
    # For now, reuse validation accuracy until attack baseline tables are joined.
    return extract_validation_accuracy(ctx)



def extract_accuracy_per_param(ctx: SeedContext) -> Optional[float]:
    acc = extract_validation_accuracy(ctx)
    params = extract_params_total(ctx)
    if acc is None or params <= 0:
        return None
    return float(acc / params)



def extract_accuracy_per_time(ctx: SeedContext) -> Optional[float]:
    acc = extract_validation_accuracy(ctx)
    duration = extract_training_time_seconds(ctx)
    if acc is None or duration is None or duration <= 0:
        return None
    return float(acc / duration)


# ============================================================
# Column registry
# ============================================================

COLUMN_EXTRACTORS: Dict[str, Callable[[SeedContext], Any]] = {
    "Family": extract_family,
    "Architecture": extract_architecture,
    "Variant Configuration (knobs)": extract_variant_configuration_knobs,
    "Seed": extract_seed,
    "Training Time (s)": extract_training_time_seconds,
    "Epoch Selected": extract_epoch_selected,
    "Selection Value": extract_selection_value,
    "Train Accuracy": extract_train_accuracy,
    "Validation Accuracy": extract_validation_accuracy,
    "Test Accuracy": extract_test_accuracy,
    "Train Loss": extract_train_loss,
    "Validation Loss": extract_validation_loss,
    "Test Loss": extract_test_loss,
    "Generalization Gap (Acc)": extract_generalization_gap_acc,
    "Generalization Gap (Loss)": extract_generalization_gap_loss,
    "Best-Worst Acc Gap": extract_best_worst_acc_gap,
    "Convergence Epoch": extract_convergence_epoch,
    "Gradient Norm Mean": extract_gradient_norm_mean,
    "Gradient Norm Std": extract_gradient_norm_std,
    "Accuracy per Param": extract_accuracy_per_param,
    "Accuracy per Time": extract_accuracy_per_time,
}


# ============================================================
# Extraction pipeline
# ============================================================


def build_seed_contexts(root_dir: Path, load_mode: str, map_location: str) -> List[SeedContext]:
    contexts: List[SeedContext] = []

    for model_dir_str in crawl_experiments(str(root_dir)):
        model_dir = Path(model_dir_str)
        relative_model_dir = os.path.relpath(model_dir, root_dir)
        hydra_config = read_hydra_config(model_dir)
        loaded = load_model(str(model_dir), best=(load_mode == "best"), map_location=map_location)

        for seed, model in loaded.items():
            seed_dir = model_dir / str(seed)
            ctx = SeedContext(
                root_dir=root_dir,
                model_dir=model_dir,
                relative_model_dir=relative_model_dir,
                seed=int(seed),
                load_mode=load_mode,
                checkpoint_path=infer_checkpoint_path(seed_dir, load_mode),
                model=model,
                hydra_config=hydra_config,
                metrics_df=read_metrics_df(seed_dir),
                model_info_text=parse_model_info(seed_dir),
            )
            contexts.append(ctx)

    return contexts



def extract_row(ctx: SeedContext) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for column, extractor in COLUMN_EXTRACTORS.items():
        try:
            row[column] = extractor(ctx)
        except Exception as exc:
            log.warning(
                "Column extraction failed | column=%s | model=%s | seed=%s | error=%s",
                column,
                ctx.relative_model_dir,
                ctx.seed,
                exc,
            )
            row[column] = None
    return row



def build_inventory_dataframe(root_dir: Path, load_mode: str = "best", map_location: str = "cpu") -> pd.DataFrame:
    contexts = build_seed_contexts(root_dir=root_dir, load_mode=load_mode, map_location=map_location)
    rows = [extract_row(ctx) for ctx in tqdm(contexts, desc="Extracting model inventory")]
    return pd.DataFrame(rows)



def save_outputs(df: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    xlsx_path = output_prefix.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        log.warning("Failed to write Excel file at %s: %s", xlsx_path, exc)

    log.info("Wrote CSV: %s", csv_path)
    log.info("Wrote XLSX: %s", xlsx_path)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    root_dir = "logs/ByModel"
    output_prefix = Path("outputs/model_inventory")
    load_mode = "best"
    map_location = "cpu"

    logging.basicConfig(format="%(levelname)s | %(message)s")

    root_dir = Path(root_dir).resolve()
    output_prefix = Path(output_prefix).resolve()

    df = build_inventory_dataframe(
        root_dir=root_dir,
        load_mode=load_mode,
        map_location=map_location,
    )
    save_outputs(df, output_prefix)


if __name__ == "__main__":
    main()
