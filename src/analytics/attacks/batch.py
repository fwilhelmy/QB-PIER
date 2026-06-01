"""Batch attack figure generation (FGSM recompute + plots) for stored runs."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from attacks import AttackFactory
from datasets import DatasetFactory
from loading import load_experiment
from analytics.paths import find_project_root, output_dir, training_log_root

from analytics.attacks import (
    cell_effectiveness_auc_drop,
    cell_effectiveness_eps_to_fraction,
    cell_effectiveness_max_rel_drop,
    cell_effectiveness_max_steepness,
    plot_attack_accuracy_df,
    plot_attack_effect_on_accuracy,
    plot_confusion_matrix,
    plot_model_attack_effectiveness_matrix,
    plot_perturbation_and_saliency,
    plot_tp_heatmaps_by_class,
    plot_transfer_attack_effects,
)
from analytics.attacks._eval_helpers import LimitedDataset, ensure_model_metadata

log = logging.getLogger(__name__)


def compute_attack_records(
    models: List[object],
    model_names: List[str],
    dataset: LimitedDataset,
    *,
    attack_epsilons: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, object, torch.Tensor]:
    """Run a compact FGSM sweep and collect attack statistics."""
    fgsm = AttackFactory.create("fgsm", dataset=dataset, batch_size=min(batch_size, len(dataset)))
    records: List[Dict[str, object]] = []
    y_true = dataset.targets

    for model in models:
        model.eval()
        gradients = fgsm.compute_gradients(model)
        if getattr(model, "log_dir", None) not in fgsm.gradients:
            fgsm.gradients[model.log_dir] = gradients

    max_eps = max(attack_epsilons)

    for attacker, attacker_name in zip(models, model_names):
        for victim, victim_name in zip(models, model_names):
            for epsilon in attack_epsilons:
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
                        "source_model": attacker_name,
                        "target_model": victim_name,
                        "epsilon": float(epsilon),
                        "accuracy": float(accuracy),
                        "preds": preds,
                    }
                )

    return pd.DataFrame.from_records(records), fgsm, y_true


def run_attack_figures_for_tree(
    model_logdir: str,
    output_dir: str,
    *,
    attack_epsilons: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    max_dataset_samples: int = 128,
    batch_size: int = 32,
    dataset_name: str = "mnist",
    dataset_root: str = "./data/processed",
) -> None:
    """Recompute FGSM transfer sweeps and refresh attack-related figures."""
    os.makedirs(output_dir, exist_ok=True)
    trained_models = load_experiment(model_logdir, load="final")

    try:
        dm = DatasetFactory.create(dataset_name, root=dataset_root, train=False)
        full_dataset = dm.dataset
        classes = getattr(full_dataset, "classes", [str(i) for i in range(10)])

        limit = min(max_dataset_samples, len(full_dataset))
        data_samples, labels = [], []
        for idx in range(limit):
            sample, label = full_dataset[idx]
            sample_t = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
            data_samples.append(sample_t)
            labels.append(int(label))

        dataset = LimitedDataset(torch.stack(tuple(data_samples)), torch.tensor(labels), classes)
    except Exception as exc:
        log.warning("Failed to load dataset %s: %s", dataset_name, exc)
        return

    first_seed_models: List[object] = []
    model_names: List[str] = []
    for model_name, model_seeds in trained_models.items():
        if not model_seeds:
            continue
        first_model = next(iter(model_seeds.values()))
        ensure_model_metadata(model_name, first_model, model_logdir)
        first_seed_models.append(first_model)
        model_names.append(model_name)

    if not first_seed_models:
        log.warning("No trained models available for analytics")
        return

    df, fgsm, y_true = compute_attack_records(
        first_seed_models,
        model_names,
        dataset,
        attack_epsilons=attack_epsilons,
        batch_size=batch_size,
    )
    if df.empty:
        log.warning("Failed to compute attack statistics")
        return

    for cell_fn, fname in (
        (cell_effectiveness_max_steepness, "attack_effectiveness_matrix_max_steepness.png"),
        (cell_effectiveness_max_rel_drop, "attack_effectiveness_matrix_max_rel_drop.png"),
        (cell_effectiveness_auc_drop, "attack_effectiveness_matrix_auc_drop.png"),
        (cell_effectiveness_eps_to_fraction, "attack_effectiveness_matrix_eps_to_fraction.png"),
    ):
        plot_model_attack_effectiveness_matrix(
            df,
            out_dir=output_dir,
            normalize=True,
            cell_fn=cell_fn,
            fname=fname,
        )

    plot_transfer_attack_effects(df, output_dir)

    epsilons = sorted(df["epsilon"].unique())
    accuracy_curves: Dict[str, List[float]] = {}
    for model in first_seed_models:
        subset = df[df["target_model"] == model.log_dir]
        stats = subset.groupby("epsilon")["accuracy"].mean().reindex(epsilons, fill_value=np.nan)
        accuracy_curves[model.log_dir] = stats.tolist()

    plot_tp_heatmaps_by_class(
        df=df,
        y_true=y_true,
        classes=dataset.classes,
        root_dir=output_dir,
        models_name=list(trained_models.keys()),
    )

    plot_attack_effect_on_accuracy(
        "plot_attack_effect_on_accuracy",
        first_seed_models,
        epsilons,
        accuracy_curves,
        output_dir,
    )

    for source_model in sorted(df["source_model"].unique()):
        plot_attack_accuracy_df(df, source_model, root_dir=output_dir)

    for model_name, model_seeds in trained_models.items():
        first_model = next(iter(model_seeds.values()))
        for epsilon in attack_epsilons:
            dirpath = os.path.join(output_dir, model_name, str(epsilon))
            plot_perturbation_and_saliency(first_model, fgsm, "cpu", epsilon, dirpath)

    for model_name in trained_models.keys():
        for (_, group) in df.loc[df["source_model"] == df["target_model"]].groupby("source_model"):
            for epsilon, subgrp in group.groupby("epsilon"):
                source_model = subgrp["source_model"]
                eps_str = str(subgrp["epsilon"].item())
                y_pred = subgrp["preds"].values
                dirpath = os.path.join(output_dir, model_name, eps_str)
                plot_confusion_matrix(
                    y_true.cpu().numpy(),
                    y_pred[0].cpu().numpy(),
                    dataset.classes,
                    root_dir=dirpath,
                    source_model=source_model,
                    normalize=True,
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    project_root = find_project_root()
    run_attack_figures_for_tree(
        model_logdir=str(training_log_root(project_root)),
        output_dir=str(output_dir(project_root, "attack_batch")),
    )


if __name__ == "__main__":
    main()
