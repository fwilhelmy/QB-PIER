from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from datasets import DatasetFactory
from loading import crawl_experiments, load_model
from visualisation.model.hquanv import is_hquanv_model, visualize_hquanv_sample

log = logging.getLogger(__name__)


def _load_run_config(model_dir: str):
    config_path = os.path.join(model_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def _load_debug_batch(model_dir: str, n_samples: int, use_train_split: bool = False):
    cfg = _load_run_config(model_dir)
    dataset_name = cfg["dataset"]["dataset"]
    dataset_root = cfg["dataset"].get("root")

    dataset_manager = DatasetFactory.create(
        dataset_name,
        root=dataset_root,
        train=use_train_split,
    )
    dataset = dataset_manager.dataset

    samples = []
    labels = []
    for idx in range(min(n_samples, len(dataset))):
        x, y = dataset[idx]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        samples.append(x)
        labels.append(int(y))

    batch = torch.stack(samples, dim=0)
    return batch, labels, cfg


def _iter_hquanv_experiments(root_dir: str):
    for model_dir in crawl_experiments(root_dir):
        try:
            models = load_model(model_dir, best=LOAD_BEST_CHECKPOINT, map_location=MAP_LOCATION)
        except Exception as exc:
            log.warning("Skipping %s because loading failed: %s", model_dir, exc)
            continue

        if not models:
            continue

        first_model = next(iter(models.values()))
        if not is_hquanv_model(first_model):
            continue

        yield model_dir, models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    experiment_root = EXPERIMENT_ROOT
    output_root = OUTPUT_ROOT
    os.makedirs(output_root, exist_ok=True)

    model_dirs = list(_iter_hquanv_experiments(experiment_root))
    if not model_dirs:
        log.warning("No HQuanv models found under %s", experiment_root)
        return

    for model_dir, models_by_seed in tqdm(model_dirs, desc="HQuanv models", unit="model"):
        rel_dir = os.path.relpath(model_dir, experiment_root)
        batch, labels, cfg = _load_debug_batch(model_dir, n_samples=N_SAMPLES, use_train_split=USE_TRAIN_SPLIT)

        for seed, model in tqdm(models_by_seed.items(), desc=f"Seeds for {rel_dir}", leave=False, unit="seed"):
            model.eval()
            seed_out_dir = os.path.join(output_root, rel_dir, str(seed))
            os.makedirs(seed_out_dir, exist_ok=True)

            visualize_hquanv_sample(
                model=model,
                inputs=batch,
                out_dir=seed_out_dir,
                prefix=SAMPLE_PREFIX,
            )

            with open(os.path.join(seed_out_dir, "labels.txt"), "w", encoding="utf-8") as handle:
                handle.write("\n".join(str(label) for label in labels))

            with open(os.path.join(seed_out_dir, "source_model_dir.txt"), "w", encoding="utf-8") as handle:
                handle.write(model_dir)

            OmegaConf.save(cfg, os.path.join(seed_out_dir, "resolved_config.yaml"))


# ---------------------------------------------------------------------------
# Debug values. Edit these directly and run the file.
# ---------------------------------------------------------------------------
EXPERIMENT_ROOT = "./logs/ByModel"
OUTPUT_ROOT = "./outputs/hquanv_visualisation"
MAP_LOCATION = "cpu"
LOAD_BEST_CHECKPOINT = True
USE_TRAIN_SPLIT = False
N_SAMPLES = 2
SAMPLE_PREFIX = "dataset_sample"


if __name__ == "__main__":
    main()
