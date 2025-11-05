"""Utility script to regenerate the attack visualisations for stored runs.

Edit :data:`MODEL_LOGDIR` to point at the directory containing the trained
models you would like to inspect. Running this module will recompute the FGSM
transfer sweeps and refresh all attack-related figures.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from LoggingService import LoggingService
from attacks import AttackFactory
from datasets import DatasetFactory
from loading import load_experiment
from visualisation.attacks import (
    plot_attack_accuracy_df,
    plot_attack_effect_on_accuracy,
    plot_confusion_matrix,
    plot_perturbation_and_saliency,
    plot_transfer_attack_effects,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User editable parameters
# ---------------------------------------------------------------------------
MODEL_LOGDIR = "REPLACE_WITH_YOUR_LOGDIR"
"""Path to the directory that contains the trained model experiment folders."""

OUTPUT_SUBDIR = "visualisation_test_outputs"
"""Relative directory (inside :data:`MODEL_LOGDIR`) where figures are stored."""

ATTACK_EPSILONS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
"""Small set of epsilon values used when recomputing FGSM transfer attacks."""

MAX_DATASET_SAMPLES = 128
"""Maximum number of dataset samples used for fast visualisation checks."""

BATCH_SIZE = 32
"""Mini-batch size employed when computing attack gradients."""

# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


class LimitedDataset(torch.utils.data.Dataset):
    """Tiny wrapper that keeps tensors and labels alongside class metadata."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, classes: Iterable[str]):
        super().__init__()
        self.data = data
        self.targets = targets
        self.classes = list(classes)

    def __len__(self) -> int:  # type: ignore[override]
        return self.data.size(0)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.data[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _ensure_model_metadata(model_name: str, model) -> None:
    """Ensure the loaded model exposes the attributes required by the plots."""

    if not hasattr(model, "model_name"):
        setattr(model, "model_name", model_name)
    if not hasattr(model, "model_path"):
        setattr(model, "model_path", getattr(model, "log_dir", model_name))
    if not hasattr(model, "root_dir"):
        setattr(model, "root_dir", getattr(model, "log_dir", MODEL_LOGDIR))


def _discover_seed_dir(seed_model) -> Path:
    log_dir = getattr(seed_model, "log_dir", None)
    if log_dir is None:
        raise RuntimeError("Loaded model does not expose a 'log_dir' attribute")
    return Path(log_dir)


def _load_dataset(seed_model) -> LimitedDataset | None:
    """Recreate the dataset referenced by the stored Hydra config."""

    try:
        seed_dir = _discover_seed_dir(seed_model)
        cfg_path = seed_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Hydra config not found at {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            raise KeyError("Dataset section missing from Hydra config")

        dataset_name = str(dataset_cfg.get("dataset", "")).lower()
        if not dataset_name:
            raise ValueError("Dataset name missing from config")

        dataset_root = dataset_cfg.get("root", "./data/processed")
        dm = DatasetFactory.create(dataset_name, root=str(dataset_root), train=False)
        full_dataset = dm.dataset

        classes = getattr(full_dataset, "classes", [str(i) for i in range(dataset_cfg.get("classes", 10))])

        limit = min(MAX_DATASET_SAMPLES, len(full_dataset))
        data_samples, labels = [], []
        for idx in range(limit):
            sample, label = full_dataset[idx]
            data_samples.append(sample)
            labels.append(int(label))

        data_tensor = torch.stack(tuple(data_samples))
        label_tensor = torch.tensor(labels)
        return LimitedDataset(data_tensor, label_tensor, classes)
    except Exception as exc:  # pragma: no cover - defensive logging only
        log.warning("Failed to recreate dataset for visualisations: %s", exc)
        return None


def _compute_attack_records(models: List[object], dataset: LimitedDataset) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, object]:
    """Run a compact FGSM sweep and collect attack statistics."""

    if not models:
        return pd.DataFrame(), torch.empty(0), torch.empty(0), None

    fgsm = AttackFactory.create("fgsm", dataset=dataset, batch_size=min(BATCH_SIZE, len(dataset)))
    records: List[Dict[str, object]] = []
    y_true = dataset.targets
    last_confusion: Tuple[torch.Tensor, torch.Tensor, object] | None = None

    for model in models:
        model.eval()
        gradients = fgsm.compute_gradients(model)
        # ensure gradients are reachable via both model_path and model_name
        if getattr(model, "model_name", None) not in fgsm.gradients:
            fgsm.gradients[model.model_name] = gradients

    max_eps = max(ATTACK_EPSILONS)

    for attacker in models:
        for victim in models:
            for epsilon in ATTACK_EPSILONS:
                adv_examples = fgsm.attack(attacker, epsilon)
                adv_examples = adv_examples.to(torch.float32)
                victim.eval()
                with torch.no_grad():
                    logits = victim(adv_examples)
                    preds = torch.argmax(logits, dim=1)
                accuracy = (preds == y_true[: preds.size(0)]).float().mean().item()
                records.append(
                    {
                        "seed": 0,
                        "source_model": attacker.model_name,
                        "target_model": victim.model_name,
                        "epsilon": float(epsilon),
                        "accuracy": float(accuracy),
                    }
                )
                if epsilon == max_eps:
                    last_confusion = (y_true[: preds.size(0)], preds, attacker)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df, torch.empty(0), torch.empty(0), fgsm

    if last_confusion is None:
        return df, torch.empty(0), torch.empty(0), fgsm

    targets_tensor, preds_tensor, attacker_model = last_confusion
    return df, targets_tensor, preds_tensor, attacker_model


def _run_attack_visualisations(trained_models: Dict[str, Dict[int, object]], dataset: LimitedDataset) -> None:
    """Produce attack-related figures using freshly computed FGSM sweeps."""

    first_seed_models = [next(iter(seeds.values())) for seeds in trained_models.values() if seeds]
    if not first_seed_models:
        log.warning("No models available for attack visualisations")
        return

    if dataset is None:
        log.warning("Dataset unavailable – attack plots require a dataset to regenerate")
        return

    df, y_true, y_pred, fgsm = _compute_attack_records(first_seed_models, dataset)
    if df.empty:
        log.warning("Failed to compute attack statistics")
        return

    log.info("Generating transfer attack visualisations")
    plot_transfer_attack_effects(df, title_prefix=f"{OUTPUT_SUBDIR}/attacks/")

    epsilons = sorted(df["epsilon"].unique())
    accuracy_curves: Dict[str, List[float]] = {}
    for model in first_seed_models:
        subset = df[df["target_model"] == model.model_name]
        stats = subset.groupby("epsilon")["accuracy"].mean().reindex(epsilons, fill_value=np.nan)
        accuracy_curves[model.model_name] = stats.tolist()

    plot_attack_effect_on_accuracy(
        f"{OUTPUT_SUBDIR}/attack_accuracy_overview",
        first_seed_models,
        epsilons,
        accuracy_curves,
    )

    for source_model in sorted(df["source_model"].unique()):
        plot_attack_accuracy_df(df, source_model, root_dir=os.path.join(OUTPUT_SUBDIR, "attacks"))

    classes = dataset.classes if hasattr(dataset, "classes") else [str(i) for i in range(int(y_true.max().item()) + 1)]
    if y_true.numel() and y_pred.numel():
        plot_confusion_matrix(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            classes,
            root_dir=os.path.join(OUTPUT_SUBDIR, "attacks"),
            source_model=first_seed_models[0].model_name,
            normalize=True,
        )

    if fgsm is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epsilon = max(ATTACK_EPSILONS)
        try:
            plot_perturbation_and_saliency(first_seed_models[0], fgsm, device, epsilon)
        except Exception as exc:  # pragma: no cover - defensive logging only
            log.warning("Failed to plot perturbation/saliency maps: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if MODEL_LOGDIR == "REPLACE_WITH_YOUR_LOGDIR":
        raise RuntimeError("Please edit MODEL_LOGDIR with the path to your logs before running this script")

    logging.basicConfig(level=logging.INFO)
    log.info("Loading experiments from %s", MODEL_LOGDIR)

    output_root = os.path.join(MODEL_LOGDIR, OUTPUT_SUBDIR)
    LoggingService(output_root)

    trained_models = load_experiment(MODEL_LOGDIR, load="final")
    for model_name, seeds in trained_models.items():
        for model in seeds.values():
            _ensure_model_metadata(model_name, model)

    dataset = None
    if trained_models:
        first_model = next(iter(next(iter(trained_models.values())).values()))
        dataset = _load_dataset(first_model)

    _run_attack_visualisations(trained_models, dataset)


if __name__ == "__main__":
    main()
