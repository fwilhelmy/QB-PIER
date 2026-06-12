"""Compute and cache FGSM input gradients + gradient-similarity pair metrics.

Run on the training cluster from the repo root:

    python scripts/compute_gradient_similarity_cache.py

Loads every trained model under MODEL_LOGDIR (blacklisted models included),
computes per-sample input gradients over N_SAMPLES test images, and writes:

    CACHE_DIR/gradients/<safe_name>.pt    float16 [N, D] per model (resumable:
                                          models with an existing .pt are skipped)
    CACHE_DIR/gradient_pair_metrics.csv   long-form pairwise similarity metrics
    CACHE_DIR/meta.json                   run metadata

Sync CACHE_DIR back to the workstation and use
notebooks/attacks/gradient_similarity.ipynb for figures/tables.
Progress and failures are reported to the Discord webhook.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from tqdm import tqdm

from attacks import AttackFactory
from datasets import DatasetFactory
from loading import load_experiment
from DiscordWebhook import DiscordWebhook
from analytics.attacks._eval_helpers import LimitedDataset, ensure_model_metadata
from analytics.attacks.gradient_similarity import (
    GRADIENTS_SUBDIR,
    compute_pair_metrics_from_cache,
    safe_model_name,
)

# --- configuration (toggle here; no argparse per AGENTS.md) ---
MODEL_LOGDIR = "logs/ByModel"
CACHE_DIR = "logs/GRADIENTS"
N_SAMPLES = 512
BATCH_SIZE = 32
DATASET = "mnist"
DATASET_ROOT = "./data/processed"
TOP_K_FRACTION = 0.1

log = logging.getLogger(__name__)


def _build_dataset() -> LimitedDataset:
    dm = DatasetFactory.create(DATASET, root=DATASET_ROOT, train=False)
    full_dataset = dm.dataset
    classes = getattr(full_dataset, "classes", [str(i) for i in range(10)])
    limit = min(N_SAMPLES, len(full_dataset))
    data_samples, labels = [], []
    for idx in range(limit):
        sample, label = full_dataset[idx]
        sample_t = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
        data_samples.append(sample_t)
        labels.append(int(label))
    return LimitedDataset(torch.stack(tuple(data_samples)), torch.tensor(labels), classes)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    discord = DiscordWebhook()
    started = time.time()

    cache_dir = _REPO_ROOT / CACHE_DIR
    grad_dir = cache_dir / GRADIENTS_SUBDIR
    grad_dir.mkdir(parents=True, exist_ok=True)

    trained_models = load_experiment(str(_REPO_ROOT / MODEL_LOGDIR), load="final")
    dataset = _build_dataset()

    discord._send(
        "Gradient similarity cache: started",
        f"Models: {len(trained_models)}\nSamples: {len(dataset)}\nCache: `{cache_dir}`",
    )

    done, skipped, failed = [], [], []
    for model_name, model_seeds in tqdm(trained_models.items(), desc="Computing gradients"):
        out_path = grad_dir / f"{safe_model_name(model_name)}.pt"
        if out_path.exists():
            skipped.append(model_name)
            continue
        try:
            if not model_seeds:
                raise RuntimeError("No seeds available")
            model = next(iter(model_seeds.values()))
            ensure_model_metadata(model_name, model, MODEL_LOGDIR)
            model.eval()

            # Fresh FGSM per model keeps the in-memory gradient cache small.
            fgsm = AttackFactory.create(
                "fgsm", dataset=dataset, batch_size=min(BATCH_SIZE, len(dataset))
            )
            gradients = fgsm.compute_gradients(model)
            grads = torch.stack([grad for _, grad in gradients]).flatten(start_dim=1)
            torch.save(
                {
                    "model_name": model_name,
                    "gradients": grads.to(torch.float16),
                    "n_samples": int(grads.shape[0]),
                    "dataset": DATASET,
                },
                out_path,
            )
            done.append(model_name)
        except Exception as exc:
            failed.append(model_name)
            log.exception("Gradient computation failed for %s", model_name)
            discord._send(
                "Gradient similarity cache: model failed",
                f"Model: `{model_name}`\nError: ```{exc}```",
            )

    pair_df = compute_pair_metrics_from_cache(cache_dir, top_k_fraction=TOP_K_FRACTION)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_logdir": MODEL_LOGDIR,
        "dataset": DATASET,
        "n_samples": len(dataset),
        "top_k_fraction": TOP_K_FRACTION,
        "models_cached": len(done) + len(skipped),
        "models_computed": len(done),
        "models_skipped_existing": len(skipped),
        "models_failed": failed,
        "pair_rows": len(pair_df),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    duration = time.time() - started
    hours, minutes = int(duration // 3600), int((duration % 3600) // 60)
    discord._send(
        "Gradient similarity cache: finished",
        f"Computed: {len(done)}, skipped (cached): {len(skipped)}, failed: {len(failed)}\n"
        f"Pair rows: {len(pair_df)}\nDuration: {hours}h {minutes}m\nCache: `{cache_dir}`",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        DiscordWebhook()._send(
            "Gradient similarity cache: crashed",
            f"```{traceback.format_exc()[-1500:]}```",
        )
        raise
