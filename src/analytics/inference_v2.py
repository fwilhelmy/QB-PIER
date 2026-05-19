from __future__ import annotations

"""Run clean inference over a directory of trained models.

This is the lightweight counterpart to ``attack.py``:
- crawl a models directory using existing ``.model`` anchors
- load either the best or final checkpoint for each seed
- run a clean forward pass on the test split
- compute test accuracy and loss
- write per-seed and per-model summary CSV files

Example
-------
python src/inference.py \
    --models-path ./logs/ByModel \
    --output-dir ./logs/inference_run \
    --dataset mnist \
    --dataset-root ./data \
    --checkpoint best \
    --device cuda
"""

from types import SimpleNamespace
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DatasetFactory
from loading import load_experiment
from utils import seed_experiment

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def _default_args() -> SimpleNamespace:
    return SimpleNamespace(
        models_path="logs/ByModel",
        output_dir="outputs/inference_v2",
        dataset="mnist",
        dataset_root="./data",
        checkpoint="best",
        device="auto",
        map_location="cpu",
        batch_size=128,
        num_workers=0,
        save_preds=False,
        save_logits=False,
        overwrite=False,
        verbose=True,
    )



def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def _safe_name(name: str) -> str:
    return (
        name.replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )



def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg



@torch.no_grad()
def _evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    save_preds: bool,
    save_logits: bool,
) -> Tuple[dict, np.ndarray | None, np.ndarray | None]:
    model = model.to(device)
    model.eval()

    total_samples = 0
    total_correct = 0
    total_loss = 0.0

    preds_chunks: List[np.ndarray] = []
    logits_chunks: List[np.ndarray] = []

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        preds = torch.argmax(logits, dim=1)

        batch_size = int(yb.size(0))
        total_samples += batch_size
        total_correct += int((preds == yb).sum().item())
        total_loss += float(loss.item())

        if save_preds:
            preds_chunks.append(preds.detach().cpu().to(torch.int64).numpy())
        if save_logits:
            logits_chunks.append(logits.detach().cpu().to(torch.float16).numpy())

    accuracy = float(total_correct / total_samples) if total_samples else float("nan")
    mean_loss = float(total_loss / total_samples) if total_samples else float("nan")

    metrics = {
        "test_accuracy": accuracy,
        "test_loss": mean_loss,
        "n_correct": int(total_correct),
        "n_samples": int(total_samples),
    }

    preds_np = np.concatenate(preds_chunks, axis=0) if save_preds and preds_chunks else None
    logits_np = np.concatenate(logits_chunks, axis=0) if save_logits and logits_chunks else None
    return metrics, preds_np, logits_np



def _load_test_dataset(dataset_name: str, dataset_root: str):
    dataset_manager = DatasetFactory.create(dataset_name, root=dataset_root)
    dataset = dataset_manager.load_dataset(train=False)
    return dataset



def _extract_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return targets.detach().cpu().to(torch.int64).numpy()
        return np.asarray(targets, dtype=np.int64)
    return np.asarray([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)



def run_inference(args: SimpleNamespace) -> None:
    device = _resolve_device(args.device)
    root = args.output_dir
    tables_dir = os.path.join(root, "tables")
    preds_root = os.path.join(root, "predictions")

    _ensure_dir(root)
    _ensure_dir(tables_dir)
    if args.save_preds or args.save_logits:
        _ensure_dir(preds_root)

    dataset = _load_test_dataset(args.dataset, args.dataset_root)
    labels = _extract_labels(dataset)
    np.save(os.path.join(root, "labels.npy"), labels)

    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    trained_models: Dict[str, Dict[int, torch.nn.Module]] = load_experiment(
        args.models_path,
        load=args.checkpoint,
        map_location=args.map_location,
    )

    n_model_variants = len(trained_models)
    n_total_seed_models = sum(len(seed_map) for seed_map in trained_models.values())
    if n_total_seed_models == 0:
        raise RuntimeError(f"No models were found under: {args.models_path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models_path": args.models_path,
        "output_dir": args.output_dir,
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "dataset_size": int(len(dataset)),
        "checkpoint_kind": args.checkpoint,
        "device": device,
        "map_location": args.map_location,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "save_preds": bool(args.save_preds),
        "save_logits": bool(args.save_logits),
        "overwrite": bool(args.overwrite),
        "num_model_variants": int(n_model_variants),
        "num_seed_models": int(n_total_seed_models),
    }
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    rows: List[dict] = []

    for model_name, seed_to_model in tqdm(sorted(trained_models.items()), desc="Models"):
        for seed, model in tqdm(sorted(seed_to_model.items()), desc=f"Seeds ({model_name})", leave=False):
            seed_experiment(int(seed))

            metrics = None
            preds_np = None
            logits_np = None

            if args.save_preds or args.save_logits:
                model_safe = _safe_name(model_name)
                seed_dir = os.path.join(preds_root, model_safe, f"seed_{seed}")
                _ensure_dir(seed_dir)
                metrics_path = os.path.join(seed_dir, "metrics.json")
                preds_path = os.path.join(seed_dir, "preds.npy")
                logits_path = os.path.join(seed_dir, "logits.npy")

                can_resume = (not args.overwrite) and os.path.exists(metrics_path)
                if args.save_preds:
                    can_resume = can_resume and os.path.exists(preds_path)
                if args.save_logits:
                    can_resume = can_resume and os.path.exists(logits_path)

                if can_resume:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                else:
                    metrics, preds_np, logits_np = _evaluate_model(
                        model,
                        dataloader,
                        device,
                        save_preds=args.save_preds,
                        save_logits=args.save_logits,
                    )
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
                    if args.save_preds and preds_np is not None:
                        np.save(preds_path, preds_np)
                    if args.save_logits and logits_np is not None:
                        np.save(logits_path, logits_np.astype(np.float16))
            else:
                metrics, _, _ = _evaluate_model(
                    model,
                    dataloader,
                    device,
                    save_preds=False,
                    save_logits=False,
                )

            row = {
                "model_name": model_name,
                "seed": int(seed),
                "checkpoint_kind": args.checkpoint,
                "device": device,
                **metrics,
            }
            rows.append(row)

    per_seed_df = pd.DataFrame(rows)
    if per_seed_df.empty:
        raise RuntimeError("Inference finished but no result rows were produced.")

    per_seed_df = per_seed_df.sort_values(["model_name", "seed"]).reset_index(drop=True)
    per_seed_path = os.path.join(tables_dir, "per_seed_results.csv")
    per_seed_df.to_csv(per_seed_path, index=False)

    summary_df = (
        per_seed_df.groupby("model_name", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            min_test_accuracy=("test_accuracy", "min"),
            max_test_accuracy=("test_accuracy", "max"),
            mean_test_loss=("test_loss", "mean"),
            std_test_loss=("test_loss", "std"),
            n_samples=("n_samples", "max"),
        )
        .sort_values(["mean_test_accuracy", "model_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_path = os.path.join(tables_dir, "model_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved per-seed results to: {per_seed_path}")
    print(f"Saved model summary to: {summary_path}")
    print(f"Evaluated {n_total_seed_models} seeded models across {n_model_variants} model variants.")



def main() -> None:
    args = _default_args()
    _configure_logging(args.verbose)
    run_inference(args)


if __name__ == "__main__":
    main()
