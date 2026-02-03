from __future__ import annotations

"""
Attack runner with reproducible artifact storage.

Key goals:
- Avoid reruns: store victim logits for every (source -> victim, epsilon).
- Keep disk sane: store only a small audit set of images (e.g., 3 per class).
- Keep analysis flexible: store per-(pair, epsilon) aggregates + baselines.

Directory layout (under the Hydra run dir):
attack_experiment/
  manifest.json
  epsilons.json
  seed_<seed>/
    labels.npy
    audit_indices.npy
    audit_clean_images_uint8.npy
    baselines/
      <victim_name_safe>/
        logits_clean.npy            # float16 [N, C]
        preds_clean.npy             # uint8   [N]
    attacks/
      source=<src_safe>__target=<tgt_safe>.npy   # float16 [E, N, C]
    attacker_audit_adv/
      source=<src_safe>.npz         # adv_images_uint8 [E, K, 28, 28]
    tables/
      aggregates.csv                # one row per (seed, source, target, epsilon)
"""

from collections import defaultdict
import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks import AttackFactory
from datasets import DatasetFactory
from loading import load_experiment
from LoggingService import LoggingService

log = logging.getLogger(__name__)


# ---------------------------
# Small, dependency-free utils
# ---------------------------

def _safe_name(name: str) -> str:
    # preserve uniqueness while avoiding nested paths
    return (
        name.replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _seed_everything(seed: int) -> None:
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # If something fails here, we still want the run to proceed
        pass


def _accuracy_from_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # rows=true, cols=pred
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    # cm: [C, C] rows=true, cols=pred
    C = cm.shape[0]
    f1s: List[float] = []
    for k in range(C):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(float(f1))
    return float(np.mean(f1s))


def _to_uint8_images(x: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor in [0,1] to uint8 images [0,255].
    Accepts [K,1,28,28] or [K,28,28] or [K,1,H,W].
    """
    with torch.no_grad():
        x = x.detach().cpu()
        x = torch.clamp(x, 0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)
        if x.dim() == 4 and x.size(1) == 1:
            x = x[:, 0, ...]
        return x.numpy()


def _hash_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(a.tobytes())
    return h.hexdigest()


@torch.no_grad()
def _forward_logits_batched(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Returns float16 logits [N, C] as numpy.
    """
    model.eval()
    outs: List[np.ndarray] = []
    N = inputs.size(0)
    for i in range(0, N, batch_size):
        xb = inputs[i : i + batch_size].to(device)
        logits = model(xb).detach().cpu().to(torch.float16).numpy()
        outs.append(logits)
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def _forward_logits_dataset(
    model: torch.nn.Module,
    dataset,
    device: str,
    batch_size: int,
    num_workers: int = 0,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    outs: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb).detach().cpu().to(torch.float16).numpy()
        outs.append(logits)
    return np.concatenate(outs, axis=0)


def _select_audit_indices(labels: np.ndarray, per_class: int = 3, num_classes: int = 10) -> np.ndarray:
    """
    Pick the first `per_class` samples for each class, deterministic given ordering.
    """
    picked: List[int] = []
    counts = [0] * num_classes
    for idx, y in enumerate(labels.tolist()):
        y = int(y)
        if 0 <= y < num_classes and counts[y] < per_class:
            picked.append(idx)
            counts[y] += 1
        if all(c >= per_class for c in counts):
            break
    return np.array(picked, dtype=np.int64)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Core experiment runner
# ---------------------------

def run_attacks_and_store(trained_models: Dict[str, Dict[int, torch.nn.Module]], cfg: DictConfig, root: str) -> None:
    # epsilons: 0..1 inclusive, step 0.05 -> 21 points
    epsilons = [i / 100 for i in range(0, 105, 5)]
    E = len(epsilons)

    # Group models by seed
    models_by_seed: Dict[int, List[Tuple[str, torch.nn.Module]]] = defaultdict(list)
    for model_name, model_seeds in trained_models.items():
        for seed, model in model_seeds.items():
            models_by_seed[int(seed)].append((model_name, model))

    # Capture config snapshot early
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

    # Attack constructor args
    attack_args = {k: v for k, v in cfg.attack.items() if k != "name"}

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hydra_run_dir": HydraConfig.get().run.dir,
        "epsilons": epsilons,
        "config_resolved": cfg_resolved,
        "notes": {
            "logits_dtype": "float16",
            "attack_logits_shape": "[E, N, C]",
            "baseline_logits_shape": "[N, C]",
            "audit_images_uint8": "stored as 0..255 uint8",
        },
    }

    _ensure_dir(root)
    with open(os.path.join(root, "epsilons.json"), "w", encoding="utf-8") as f:
        json.dump({"epsilons": epsilons}, f, indent=2)

    # We'll write aggregates as we go (appendable)
    # If a previous aggregates.csv exists, load it to skip completed work.
    aggregates_path = os.path.join(root, "tables", "aggregates.csv")
    _ensure_dir(os.path.dirname(aggregates_path))
    existing_done = set()
    if os.path.exists(aggregates_path):
        try:
            prev = pd.read_csv(aggregates_path)
            for _, r in prev.iterrows():
                existing_done.add((int(r["seed"]), r["source_model"], r["target_model"]))
            log.info("Loaded aggregates.csv with %d rows, %d unique pairs.", len(prev), len(existing_done))
        except Exception as ex:
            log.warning("Could not read existing aggregates.csv (%s). Will recompute pairs.", ex)

    # Run per seed
    for seed, models in models_by_seed.items():
        log.info("=== Seed %s ===", seed)
        _seed_everything(seed)

        seed_dir = os.path.join(root, f"seed_{seed}")
        baselines_dir = os.path.join(seed_dir, "baselines")
        attacks_dir = os.path.join(seed_dir, "attacks")
        audit_adv_dir = os.path.join(seed_dir, "attacker_audit_adv")
        tables_dir = os.path.join(seed_dir, "tables")
        _ensure_dir(seed_dir)
        _ensure_dir(baselines_dir)
        _ensure_dir(attacks_dir)
        _ensure_dir(audit_adv_dir)
        _ensure_dir(tables_dir)

        # Dataset
        dm = DatasetFactory.create(cfg.dataset.dataset, root=cfg.dataset.root)
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset

        # Labels
        labels = np.array(getattr(dataset, "targets", None) if hasattr(dataset, "targets") else [dataset[i][1] for i in range(len(dataset))], dtype=np.int64)
        np.save(os.path.join(seed_dir, "labels.npy"), labels)

        N = int(labels.shape[0])
        num_classes = int(getattr(cfg.dataset, "num_classes", 10))

        # Audit indices + clean images (3 per class by default)
        audit_k = int(getattr(cfg.attack, "audit_per_class", 3))
        audit_indices = _select_audit_indices(labels, per_class=audit_k, num_classes=num_classes)
        np.save(os.path.join(seed_dir, "audit_indices.npy"), audit_indices)

        # Try to fetch clean images efficiently
        audit_clean_path = os.path.join(seed_dir, "audit_clean_images_uint8.npy")
        if not os.path.exists(audit_clean_path):
            if hasattr(dataset, "data"):
                data = dataset.data
                # torchvision MNIST: uint8 [N, 28, 28]
                if isinstance(data, torch.Tensor):
                    clean_imgs = data[audit_indices].cpu().numpy()
                    if clean_imgs.dtype != np.uint8:
                        # assume 0..1 float
                        clean_imgs = np.clip(clean_imgs * 255.0, 0, 255).round().astype(np.uint8)
                else:
                    clean_imgs = np.asarray(data)[audit_indices]
                    if clean_imgs.dtype != np.uint8:
                        clean_imgs = np.clip(clean_imgs * 255.0, 0, 255).round().astype(np.uint8)
                np.save(audit_clean_path, clean_imgs)
            else:
                # fallback: index dataset
                xs = []
                for idx in audit_indices.tolist():
                    x, _ = dataset[idx]
                    if isinstance(x, torch.Tensor):
                        xs.append(_to_uint8_images(x.unsqueeze(0))[0])
                    else:
                        arr = np.asarray(x)
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
                        xs.append(arr)
                np.save(audit_clean_path, np.stack(xs, axis=0))

        # Prepare attack object (needs dm.dataset)
        attack = AttackFactory.create(cfg.attack.name, dataset=dm.dataset, **attack_args)

        device = cfg.device.device
        batch_size = int(getattr(cfg.attack, "eval_batch_size", 512))

        # -----------------
        # Baselines (clean)
        # -----------------
        baseline_preds: Dict[str, np.ndarray] = {}
        for victim_name, victim_model in tqdm(models, desc="Baselines (clean logits)", leave=False):
            v_safe = _safe_name(victim_name)
            v_dir = os.path.join(baselines_dir, v_safe)
            _ensure_dir(v_dir)
            logits_path = os.path.join(v_dir, "logits_clean.npy")
            preds_path = os.path.join(v_dir, "preds_clean.npy")

            if os.path.exists(logits_path) and os.path.exists(preds_path):
                preds = np.load(preds_path)
                baseline_preds[victim_name] = preds
                continue

            victim_model.eval()
            logits_clean = _forward_logits_dataset(victim_model, dataset, device=device, batch_size=batch_size)
            preds_clean = np.argmax(logits_clean, axis=1).astype(np.uint8)

            np.save(logits_path, logits_clean.astype(np.float16))
            np.save(preds_path, preds_clean)
            baseline_preds[victim_name] = preds_clean

        # -----------------
        # Transfer attacks
        # -----------------
        # We'll write logits_adv incrementally to .npy files using open_memmap.
        for attacker_name, attacker_model in tqdm(models, desc="Attacker models"):
            a_safe = _safe_name(attacker_name)
            attacker_model.eval()

            # Audit storage for this attacker: [E, K, 28, 28] uint8
            audit_adv = np.zeros((E, audit_indices.shape[0], 28, 28), dtype=np.uint8)

            # Prepare output memmaps for each victim (avoid reruns; write eps slices as computed)
            victim_memmaps = {}
            victims_to_run: List[Tuple[str, torch.nn.Module]] = []
            for victim_name, victim_model in models:
                # allow skipping full attacker->victim if already complete
                pair_done_key = (seed, attacker_name, victim_name)
                out_path = os.path.join(attacks_dir, f"source={a_safe}__target={_safe_name(victim_name)}.npy")
                if pair_done_key in existing_done and os.path.exists(out_path):
                    continue
                victims_to_run.append((victim_name, victim_model))

                # Create / overwrite if partial (safer than attempting partial resumes without checks)
                mm = np.lib.format.open_memmap(
                    out_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(E, N, num_classes),
                )
                victim_memmaps[victim_name] = (mm, out_path)

            if not victims_to_run:
                # still may want audit images if missing
                audit_out = os.path.join(audit_adv_dir, f"source={a_safe}.npz")
                if not os.path.exists(audit_out):
                    # We do not have adv examples without recomputing; skip quietly.
                    pass
                continue

            # Epsilon loop: generate adv examples once per epsilon, reuse across all victims
            for e_idx, eps in enumerate(tqdm(epsilons, desc=f"Epsilons ({attacker_name})", leave=False)):
                adv_examples = attack.attack(attacker_model, eps).to(device)

                # Store audit images for this epsilon
                try:
                    audit_adv[e_idx] = _to_uint8_images(adv_examples[audit_indices])
                except Exception:
                    # If indexing fails for some reason, skip audit images
                    pass

                # Victim loop
                for victim_name, victim_model in victims_to_run:
                    logits_adv = _forward_logits_batched(victim_model, adv_examples, device=device, batch_size=batch_size)
                    victim_memmaps[victim_name][0][e_idx] = logits_adv.astype(np.float16)

                    # Aggregates row
                    preds_adv = np.argmax(logits_adv, axis=1).astype(np.uint8)
                    preds_clean = baseline_preds[victim_name]
                    cm = _confusion_matrix(preds_adv, labels, num_classes)
                    row = {
                        "seed": seed,
                        "source_model": attacker_name,
                        "target_model": victim_name,
                        "epsilon": float(eps),
                        "robust_accuracy": _accuracy_from_preds(preds_adv, labels),
                        "flip_rate": float((preds_adv != preds_clean).mean()),
                        "macro_f1": _macro_f1_from_cm(cm),
                        "n_samples": int(N),
                    }

                    # Append to global aggregates file incrementally (safe for long runs)
                    pd.DataFrame([row]).to_csv(
                        aggregates_path,
                        mode="a",
                        header=not os.path.exists(aggregates_path) or os.path.getsize(aggregates_path) == 0,
                        index=False,
                    )

                # Free memory
                del adv_examples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Flush memmaps
            for victim_name, (mm, out_path) in victim_memmaps.items():
                mm.flush()
                existing_done.add((seed, attacker_name, victim_name))

            # Save attacker audit adv images
            np.savez_compressed(
                os.path.join(audit_adv_dir, f"source={a_safe}.npz"),
                epsilons=np.array(epsilons, dtype=np.float32),
                audit_indices=audit_indices,
                adv_images_uint8=audit_adv,
            )

        # Seed-level quick summary table (optional)
        try:
            seed_agg = pd.read_csv(aggregates_path)
            seed_agg = seed_agg[seed_agg["seed"] == seed]
            seed_agg.to_csv(os.path.join(tables_dir, "aggregates_seed.csv"), index=False)
        except Exception:
            pass

    # Write manifest last (so it exists even if run is partial)
    # Add dataset label hash to allow verifying consistent ordering later.
    try:
        # Use last computed labels if available in scope; if not, omit.
        pass
    except Exception:
        pass

    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


@hydra.main(config_path="../conf", config_name="attack", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().run.dir
    ls = LoggingService(run_dir)

    # Root for this experiment
    root = os.path.join(ls.root_dir, "attack_experiment")

    models = load_experiment(
        cfg.models_path,
        load=cfg.model_load,
        map_location=cfg.device.device,
    )

    run_attacks_and_store(models, cfg, root=root)


if __name__ == "__main__":
    main()
