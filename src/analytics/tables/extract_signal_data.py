from __future__ import annotations

from types import SimpleNamespace
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DatasetFactory
from loading import crawl_experiments
from utils import seed_experiment

from analytics.tables._experiment_io import (
    find_checkpoint,
    iter_seed_dirs,
    load_single_model,
    read_hydra_config,
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    return (
        str(name)
        .replace("\\", "__")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)  # type: ignore[return-value]
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


def _safe_get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _dataset_cfg_view(hydra_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the most likely dataset config view.

    In your pipeline the fully composed Hydra config usually stores dataset
    values under cfg.dataset.*, but some experiment folders may only expose the
    composed keys at top-level or only preserve Hydra defaults metadata.
    """
    if not isinstance(hydra_cfg, dict):
        return {}
    ds = hydra_cfg.get("dataset")
    if isinstance(ds, dict):
        return ds
    return hydra_cfg


def _resolve_dataset_name(hydra_cfg: Dict[str, Any]) -> Optional[str]:
    dataset_cfg = _dataset_cfg_view(hydra_cfg)
    name = dataset_cfg.get("dataset") if isinstance(dataset_cfg, dict) else None
    if isinstance(name, str) and name.strip():
        return name

    # Fall back to Hydra defaults such as defaults: [{dataset: mnist}, ...]
    defaults = hydra_cfg.get("defaults", []) if isinstance(hydra_cfg, dict) else []
    if isinstance(defaults, list):
        for item in defaults:
            if isinstance(item, dict) and "dataset" in item:
                candidate = item.get("dataset")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate

    return None


def build_test_loader(hydra_cfg: Dict[str, Any], batch_size_override: Optional[int], num_workers_override: Optional[int]) -> tuple[DataLoader, np.ndarray]:
    dataset_cfg = _dataset_cfg_view(hydra_cfg)
    device_cfg = hydra_cfg.get("device", {}) if isinstance(hydra_cfg, dict) and isinstance(hydra_cfg.get("device", {}), dict) else {}

    dataset_name = _resolve_dataset_name(hydra_cfg)
    if not dataset_name:
        raise KeyError(
            "Could not resolve dataset name from Hydra config. "
            "Tried cfg.dataset.dataset, top-level cfg.dataset, and Hydra defaults entries."
        )

    root = dataset_cfg.get("root") if isinstance(dataset_cfg, dict) else None
    if root is None and isinstance(hydra_cfg, dict):
        data_dir = hydra_cfg.get("DATA_DIR")
        if isinstance(data_dir, str) and data_dir.strip():
            root = data_dir

    dm = DatasetFactory.create(dataset_name, root=root)
    dataset = dm.load_dataset(train=False)
    dm.dataset = dataset

    eval_bs = dataset_cfg.get("eval_batch_size") if isinstance(dataset_cfg, dict) else None
    train_bs = dataset_cfg.get("train_batch_size") if isinstance(dataset_cfg, dict) else None
    batch_size = int(batch_size_override or eval_bs or train_bs or 512)
    num_workers = int(num_workers_override if num_workers_override is not None else device_cfg.get("num_workers", 0))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets, dtype=np.int64)
    else:
        labels = np.asarray([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)

    return loader, labels


# -----------------------------------------------------------------------------
# Model inspection
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisPlan:
    analysis_module_path: Optional[str]
    analysis_stage_name: Optional[str]
    notes: str


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    modules = dict(model.named_modules())
    if path not in modules:
        raise KeyError(f"Could not find module path '{path}'. Available examples: {list(modules)[:20]}")
    return modules[path]


def build_analysis_plan(model: nn.Module) -> AnalysisPlan:
    if hasattr(model, "encoder") and model.encoder.__class__.__name__.lower() == "quanvolution":
        return AnalysisPlan(
            analysis_module_path="encoder",
            analysis_stage_name="encoder_quantum_output",
            notes="Ablates the full quantum encoder output handed to the downstream classifier.",
        )

    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        if hasattr(classifier, "qlayer"):
            return AnalysisPlan(
                analysis_module_path="classifier.qlayer",
                analysis_stage_name="classifier_quantum_output",
                notes="Ablates the quantum layer output before the post-quantum classical head.",
            )
        if hasattr(classifier, "vqc"):
            return AnalysisPlan(
                analysis_module_path="classifier.vqc",
                analysis_stage_name="classifier_quantum_output",
                notes="Ablates the VQC output before any post-quantum classical projection.",
            )

    if hasattr(model, "vqc"):
        return AnalysisPlan(
            analysis_module_path="vqc",
            analysis_stage_name="model_quantum_output",
            notes="Ablates the model-level VQC output.",
        )

    return AnalysisPlan(None, None, "No quantum analysis module detected.")


def build_param_group_prefixes(model: nn.Module) -> Dict[str, List[str]]:
    prefixes: Dict[str, List[str]] = defaultdict(list)

    if hasattr(model, "encoder"):
        prefixes["classical_encoder"].append("encoder")

    if hasattr(model, "classifier"):
        prefixes["classical_head"].append("classifier")

    encoder = getattr(model, "encoder", None)
    classifier = getattr(model, "classifier", None)

    if encoder is not None and hasattr(encoder, "quanv"):
        prefixes["quantum_core"].append("encoder.quanv.vqc")
        prefixes["quantum_readout"].append("encoder.quanv.readout")

    if classifier is not None:
        if hasattr(classifier, "fc1"):
            prefixes["pre_quantum_classical"].append("classifier.fc1")
        if hasattr(classifier, "bn"):
            prefixes["pre_quantum_classical"].append("classifier.bn")
        if hasattr(classifier, "qlayer"):
            prefixes["quantum_core"].append("classifier.qlayer")
        if hasattr(classifier, "vqc"):
            prefixes["quantum_core"].append("classifier.vqc")
        if hasattr(classifier, "fc2"):
            prefixes["post_quantum_classical"].append("classifier.fc2")
        if hasattr(classifier, "fc_out"):
            prefixes["post_quantum_classical"].append("classifier.fc_out")
        if hasattr(classifier, "post"):
            prefixes["post_quantum_classical"].append("classifier.post")
        if hasattr(classifier, "net"):
            prefixes["post_quantum_classical"].append("classifier.net")
        if hasattr(classifier, "residual_proj"):
            prefixes["post_quantum_classical"].append("classifier.residual_proj")
        if hasattr(classifier, "residual_scale"):
            prefixes["post_quantum_classical"].append("classifier.residual_scale")

    # Deduplicate while preserving order.
    deduped: Dict[str, List[str]] = {}
    for group, values in prefixes.items():
        seen: set[str] = set()
        clean: List[str] = []
        for value in values:
            if value not in seen:
                clean.append(value)
                seen.add(value)
        deduped[group] = clean
    return deduped


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def assign_parameter_groups(model: nn.Module) -> Dict[str, List[tuple[str, nn.Parameter]]]:
    prefixes = build_param_group_prefixes(model)

    ordered_prefixes: List[tuple[str, str]] = []
    for group, group_prefixes in prefixes.items():
        for prefix in group_prefixes:
            ordered_prefixes.append((group, prefix))
    ordered_prefixes.sort(key=lambda item: len(item[1]), reverse=True)

    groups: Dict[str, List[tuple[str, nn.Parameter]]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        assigned_group: Optional[str] = None
        for group, prefix in ordered_prefixes:
            if _matches_prefix(name, prefix):
                assigned_group = group
                break

        if assigned_group is None:
            if name.startswith("encoder."):
                assigned_group = "classical_encoder"
            elif name.startswith("classifier."):
                assigned_group = "classical_head"
            else:
                assigned_group = "other"

        groups[assigned_group].append((name, param))

    return dict(groups)


# -----------------------------------------------------------------------------
# Math / metrics helpers
# -----------------------------------------------------------------------------


def _flatten_batch(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def _safe_probs(logits: np.ndarray) -> np.ndarray:
    # Works whether logits are raw logits or already log-probabilities.
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _effective_rank(features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    p = (s ** 2)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float(np.exp(entropy))


def _nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        return float("nan")

    classes = np.unique(y)
    centroids = []
    centroid_labels = []
    for cls in classes:
        cls_x = x[y == cls]
        if cls_x.size == 0:
            continue
        centroids.append(cls_x.mean(axis=0))
        centroid_labels.append(int(cls))
    if not centroids:
        return float("nan")

    C = np.stack(centroids, axis=0)
    dists = ((x[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    preds = np.asarray([centroid_labels[i] for i in np.argmin(dists, axis=1)], dtype=np.int64)
    return float((preds == y).mean())


def _fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return float("nan")

    global_mean = x.mean(axis=0)
    classes = np.unique(y)
    between = 0.0
    within = 0.0
    for cls in classes:
        cls_x = x[y == cls]
        if cls_x.shape[0] == 0:
            continue
        cls_mean = cls_x.mean(axis=0)
        between += float(cls_x.shape[0] * np.sum((cls_mean - global_mean) ** 2))
        within += float(np.sum((cls_x - cls_mean) ** 2))
    if within <= 0.0:
        return float("nan")
    return float(between / within)


def summarise_representation(features: Optional[np.ndarray], labels: np.ndarray, stage: str, model_name: str, seed: int) -> List[Dict[str, Any]]:
    if features is None:
        return []
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    feature_std = np.std(x, axis=0)
    total_var = np.var(x, axis=0).sum()
    return [{
        "model": model_name,
        "seed": seed,
        "stage": stage,
        "n_samples": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "mean_feature_std": float(np.mean(feature_std)) if feature_std.size else float("nan"),
        "total_variance": float(total_var),
        "effective_rank": _effective_rank(x),
        "nearest_centroid_acc": _nearest_centroid_accuracy(x, labels),
        "fisher_ratio": _fisher_ratio(x, labels),
    }]


# -----------------------------------------------------------------------------
# Clean forward pass + activation capture
# -----------------------------------------------------------------------------


def collect_clean_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    analysis_module: Optional[nn.Module],
) -> Dict[str, Any]:
    logits_parts: List[np.ndarray] = []
    q_inputs: List[np.ndarray] = []
    q_outputs_raw: List[np.ndarray] = []
    q_outputs_flat: List[np.ndarray] = []

    handles: List[Any] = []
    if analysis_module is not None:
        def _pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            tensor = inputs[0]
            if isinstance(tensor, torch.Tensor):
                q_inputs.append(_flatten_batch(tensor))

        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            out_cpu = output.detach().cpu().float().numpy()
            q_outputs_raw.append(out_cpu)
            q_outputs_flat.append(_flatten_batch(out_cpu))

        handles.append(analysis_module.register_forward_pre_hook(_pre_hook))
        handles.append(analysis_module.register_forward_hook(_hook))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Clean inference", leave=False):
            x, _ = batch
            x = x.to(device)
            logits = model(x)
            logits_parts.append(logits.detach().cpu().float().numpy())

    for handle in handles:
        handle.remove()

    logits = np.concatenate(logits_parts, axis=0) if logits_parts else np.empty((0, 0), dtype=np.float32)
    q_in = np.concatenate(q_inputs, axis=0) if q_inputs else None
    q_out_raw = np.concatenate(q_outputs_raw, axis=0) if q_outputs_raw else None
    q_out_flat = np.concatenate(q_outputs_flat, axis=0) if q_outputs_flat else None

    return {
        "logits": logits,
        "q_input": q_in,
        "q_output_raw": q_out_raw,
        "q_output_flat": q_out_flat,
    }


# -----------------------------------------------------------------------------
# Gradient norms on clean loss
# -----------------------------------------------------------------------------


def collect_block_gradient_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    param_groups: Dict[str, List[tuple[str, nn.Parameter]]],
    max_batches: Optional[int],
) -> List[Dict[str, Any]]:
    batch_metrics: Dict[str, List[float]] = defaultdict(list)
    batch_losses: List[float] = []

    model.eval()
    loss_fn = getattr(model, "loss_fn", torch.nn.CrossEntropyLoss())

    iterator = enumerate(loader)
    for batch_idx, batch in tqdm(iterator, desc="Gradient norms", leave=False):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        batch_losses.append(float(loss.detach().cpu().item()))

        for group_name, params in param_groups.items():
            sq_sum = 0.0
            for _, param in params:
                grad = param.grad
                if grad is None:
                    continue
                sq_sum += float(torch.sum(grad.detach().float() ** 2).item())
            batch_metrics[group_name].append(math.sqrt(sq_sum))

    rows: List[Dict[str, Any]] = []
    for group_name, params in param_groups.items():
        param_sq = 0.0
        param_count = 0
        for _, param in params:
            tensor = param.detach().float()
            param_sq += float(torch.sum(tensor ** 2).item())
            param_count += int(param.numel())
        param_l2 = math.sqrt(param_sq)

        values = batch_metrics.get(group_name, [])
        rows.append({
            "group": group_name,
            "param_count": param_count,
            "param_l2": param_l2,
            "grad_l2_mean": float(np.mean(values)) if values else float("nan"),
            "grad_l2_std": float(np.std(values)) if values else float("nan"),
            "grad_l2_max": float(np.max(values)) if values else float("nan"),
            "relative_grad_mean": float(np.mean(values) / max(param_l2, 1e-12)) if values else float("nan"),
            "n_grad_batches": len(values),
            "mean_loss_used_for_grad": float(np.mean(batch_losses)) if batch_losses else float("nan"),
        })

    total_params = sum(row["param_count"] for row in rows)
    for row in rows:
        row["param_pct"] = float(row["param_count"] / total_params) if total_params else float("nan")

    return rows


# -----------------------------------------------------------------------------
# Ablation evaluation
# -----------------------------------------------------------------------------


def run_output_ablation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    analysis_module: nn.Module,
    baseline_logits: np.ndarray,
    baseline_output_raw: np.ndarray,
    mode: str,
) -> Dict[str, Any]:
    if mode not in {"zero", "mean", "shuffle"}:
        raise ValueError(f"Unsupported ablation mode: {mode}")

    outputs = np.asarray(baseline_output_raw)
    mean_output = outputs.mean(axis=0, keepdims=False)
    perm = np.random.default_rng(42).permutation(outputs.shape[0]) if mode == "shuffle" else None

    logits_parts: List[np.ndarray] = []
    cursor = 0

    def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal cursor
        if not isinstance(output, torch.Tensor):
            return output

        batch_size = int(output.shape[0])
        if mode == "zero":
            replacement = torch.zeros_like(output)
        elif mode == "mean":
            replacement = torch.as_tensor(mean_output, dtype=output.dtype, device=output.device)
            replacement = replacement.unsqueeze(0).expand_as(output)
        else:
            assert perm is not None
            replacement_np = outputs[perm[cursor: cursor + batch_size]]
            replacement = torch.as_tensor(replacement_np, dtype=output.dtype, device=output.device)
        cursor += batch_size
        return replacement

    handle = analysis_module.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Ablation={mode}", leave=False):
            x, _ = batch
            x = x.to(device)
            logits = model(x)
            logits_parts.append(logits.detach().cpu().float().numpy())
    handle.remove()

    logits = np.concatenate(logits_parts, axis=0) if logits_parts else np.empty_like(baseline_logits)
    preds = np.argmax(logits, axis=1)
    base_preds = np.argmax(baseline_logits, axis=1)
    mean_logit_l2 = float(np.mean(np.linalg.norm(logits - baseline_logits, axis=1))) if logits.size else float("nan")
    pred_change_rate = float(np.mean(preds != base_preds)) if preds.size else float("nan")

    base_probs = _safe_probs(baseline_logits)
    probs = _safe_probs(logits)
    kl = np.sum(base_probs * (np.log(np.clip(base_probs, 1e-12, None)) - np.log(np.clip(probs, 1e-12, None))), axis=1)

    return {
        "mode": mode,
        "logits": logits,
        "mean_logit_l2": mean_logit_l2,
        "pred_change_rate": pred_change_rate,
        "mean_kl_from_baseline": float(np.mean(kl)) if kl.size else float("nan"),
    }


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def save_table(df: pd.DataFrame, path_no_suffix: Path) -> Path:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path = path_no_suffix.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        out_path = path_no_suffix.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        return out_path


def analyze_model_seed(
    model_name: str,
    model: nn.Module,
    hydra_cfg: Dict[str, Any],
    seed: int,
    output_dir: Path,
    batch_size_override: Optional[int],
    num_workers_override: Optional[int],
    grad_batches: Optional[int],
    device: torch.device,
) -> Dict[str, pd.DataFrame]:
    model = model.to(device)
    seed_experiment(seed)

    loader, labels = build_test_loader(hydra_cfg, batch_size_override=batch_size_override, num_workers_override=num_workers_override)
    plan = build_analysis_plan(model)
    analysis_module = get_module_by_path(model, plan.analysis_module_path) if plan.analysis_module_path else None

    clean = collect_clean_pass(model, loader, device, analysis_module)
    baseline_logits = clean["logits"]
    baseline_preds = np.argmax(baseline_logits, axis=1) if baseline_logits.size else np.asarray([], dtype=np.int64)

    baseline_row = pd.DataFrame([{
        "model": model_name,
        "seed": seed,
        "analysis_module": plan.analysis_module_path,
        "analysis_stage": plan.analysis_stage_name,
        "analysis_notes": plan.notes,
        "n_samples": int(labels.shape[0]),
        "clean_accuracy": float(np.mean(baseline_preds == labels)) if baseline_preds.size else float("nan"),
        "clean_n_classes": int(baseline_logits.shape[1]) if baseline_logits.ndim == 2 else None,
    }])

    param_groups = assign_parameter_groups(model)
    grad_rows = collect_block_gradient_stats(model, loader, device, param_groups, grad_batches)
    block_rows = pd.DataFrame([{**row, "model": model_name, "seed": seed} for row in grad_rows])

    rep_rows: List[Dict[str, Any]] = []
    rep_rows.extend(summarise_representation(clean.get("q_input"), labels, "quantum_input", model_name, seed))
    rep_rows.extend(summarise_representation(clean.get("q_output_flat"), labels, "quantum_output", model_name, seed))
    representation_rows = pd.DataFrame(rep_rows)

    interface_rows: List[Dict[str, Any]] = []
    q_in = clean.get("q_input")
    q_out = clean.get("q_output_flat")
    if q_in is not None or q_out is not None:
        interface_rows.append({
            "model": model_name,
            "seed": seed,
            "analysis_module": plan.analysis_module_path,
            "input_dim": int(q_in.shape[1]) if q_in is not None else None,
            "output_dim": int(q_out.shape[1]) if q_out is not None else None,
            "width_ratio_out_over_in": float(q_out.shape[1] / q_in.shape[1]) if (q_in is not None and q_out is not None and q_in.shape[1] > 0) else float("nan"),
            "effective_rank_in": _effective_rank(q_in) if q_in is not None else float("nan"),
            "effective_rank_out": _effective_rank(q_out) if q_out is not None else float("nan"),
            "nearest_centroid_in": _nearest_centroid_accuracy(q_in, labels) if q_in is not None else float("nan"),
            "nearest_centroid_out": _nearest_centroid_accuracy(q_out, labels) if q_out is not None else float("nan"),
            "fisher_ratio_in": _fisher_ratio(q_in, labels) if q_in is not None else float("nan"),
            "fisher_ratio_out": _fisher_ratio(q_out, labels) if q_out is not None else float("nan"),
        })
    interface_rows_df = pd.DataFrame(interface_rows)

    ablation_rows: List[Dict[str, Any]] = []
    analysis_seed_dir = output_dir / f"seed_{seed}" / _safe_name(model_name)
    _ensure_dir(analysis_seed_dir)
    np.save(analysis_seed_dir / "clean_logits.npy", baseline_logits.astype(np.float32))
    np.save(analysis_seed_dir / "labels.npy", labels.astype(np.int64))
    if q_out is not None:
        np.save(analysis_seed_dir / "quantum_output_flat.npy", q_out.astype(np.float32))
    if q_in is not None:
        np.save(analysis_seed_dir / "quantum_input_flat.npy", q_in.astype(np.float32))

    if analysis_module is not None and clean.get("q_output_raw") is not None:
        for mode in ("zero", "mean", "shuffle"):
            result = run_output_ablation(
                model=model,
                loader=loader,
                device=device,
                analysis_module=analysis_module,
                baseline_logits=baseline_logits,
                baseline_output_raw=clean["q_output_raw"],
                mode=mode,
            )
            preds = np.argmax(result["logits"], axis=1) if result["logits"].size else np.asarray([], dtype=np.int64)
            np.save(analysis_seed_dir / f"ablation_{mode}_logits.npy", result["logits"].astype(np.float32))
            ablation_rows.append({
                "model": model_name,
                "seed": seed,
                "mode": mode,
                "accuracy": float(np.mean(preds == labels)) if preds.size else float("nan"),
                "accuracy_drop": float(np.mean(baseline_preds == labels) - np.mean(preds == labels)) if preds.size else float("nan"),
                "pred_change_rate": result["pred_change_rate"],
                "mean_logit_l2": result["mean_logit_l2"],
                "mean_kl_from_baseline": result["mean_kl_from_baseline"],
            })
    ablation_rows_df = pd.DataFrame(ablation_rows)

    return {
        "baseline": baseline_row,
        "blocks": block_rows,
        "representations": representation_rows,
        "interface": interface_rows_df,
        "ablations": ablation_rows_df,
    }


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
        root="logs/ByModel",
        output_dir="outputs/signal_experiment",
        load="best",
        device="cpu",
        batch_size=None,
        num_workers=None,
        grad_batches=None,
        model_filter=None,
    )


def main() -> None:
    args = default_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "signal_experiment"
    _ensure_dir(output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "output_dir": str(output_dir),
        "load_mode": args.load,
        "device": args.device,
        "batch_size_override": args.batch_size,
        "num_workers_override": args.num_workers,
        "grad_batches": args.grad_batches,
        "notes": {
            "gradient_norms": "Computed post-hoc on clean test loss, not from training-time optimizer steps.",
            "update_norms": "Not extracted because the current logging contract does not guarantee full checkpoint trajectories across epochs.",
            "ablations": ["zero", "mean", "shuffle"],
            "representation_metrics": ["effective_rank", "nearest_centroid_acc", "fisher_ratio"],
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    model_dirs = [Path(p) for p in crawl_experiments(str(root))]
    if args.model_filter:
        model_dirs = [p for p in model_dirs if args.model_filter.lower() in os.path.relpath(p, root).lower()]

    all_tables: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    device = torch.device(args.device)

    for model_dir in tqdm(model_dirs, desc="Models"):
        relative_model = os.path.relpath(model_dir, root)
        hydra_cfg = {}

        for seed_dir in tqdm(list(iter_seed_dirs(model_dir)), desc=f"Seeds ({relative_model})", leave=False):
            checkpoint = find_checkpoint(seed_dir, args.load)
            if checkpoint is None:
                log.warning("Skipping %s (no checkpoint found)", seed_dir)
                continue

            try:
                model = load_single_model(model_dir, seed_dir, checkpoint, map_location=args.device)
                hydra_cfg = read_hydra_config(model_dir, seed_dir)
                tables = analyze_model_seed(
                    model_name=relative_model,
                    model=model,
                    hydra_cfg=hydra_cfg,
                    seed=int(seed_dir.name),
                    output_dir=output_dir,
                    batch_size_override=args.batch_size,
                    num_workers_override=args.num_workers,
                    grad_batches=args.grad_batches,
                    device=device,
                )
                for key, df in tables.items():
                    if not df.empty:
                        all_tables[key].append(df)
            except Exception as exc:
                log.exception("Failed analysis for %s seed=%s: %s", relative_model, seed_dir.name, exc)

    saved_paths: Dict[str, str] = {}
    for key, frames in all_tables.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        saved = save_table(df, output_dir / "tables" / key)
        saved_paths[key] = str(saved)
        log.info("Saved %s rows to %s", len(df), saved)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        manifest["tables"] = saved_paths
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
