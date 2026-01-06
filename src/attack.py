"""Tools for running transfer attacks between trained models.

The module organises attack execution, logging, and artefact persistence
into focused helper functions for easier maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import logging

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from attacks import AttackFactory, AttackResult, AttackRepository
from datasets import DatasetFactory
from LoggingService import LoggingService
from utils import calculate_accuracy, seed_experiment
from visualisation.attacks import plot_transfer_attack_effects

log = logging.getLogger(__name__)

def _group_models_by_seed(trained_models: Dict[str, Dict[int, torch.nn.Module]]) -> Dict[int, List[Tuple[str, torch.nn.Module]]]:
    models_by_seed: Dict[int, List[Tuple[str, torch.nn.Module]]] = {}
    for model_name, model_seeds in trained_models.items():
        for seed, model in model_seeds.items():
            models_by_seed.setdefault(seed, []).append((model_name, model))
    return models_by_seed


def _prepare_attack(cfg: DictConfig):
    dm = DatasetFactory.create(cfg.dataset.dataset, root=cfg.dataset.root)
    dataset = dm.load_dataset(train=False)
    dm.dataset = dataset
    attack_args = {k: v for k, v in cfg.attack.items() if k != "name"}
    attack = AttackFactory.create(cfg.attack.name, dataset=dm.dataset, **attack_args)
    return dataset, attack


def _log_tensorboard(
    writer: SummaryWriter,
    result: AttackResult,
) -> None:
    tag = f"{result.source_model}->{result.target_model}/accuracy"
    writer.add_scalar(tag, result.accuracy, global_step=int(result.epsilon * 100))


def _evaluate_attack(
    *,
    attack,
    attacker_model,
    victim_model,
    epsilon: float,
    targets: Sequence[int],
    device: str,
) -> Tuple[Tensor, Tensor, float]:
    adv_examples = attack.attack(attacker_model, epsilon).to(device)
    victim_model.eval()
    with torch.no_grad():
        adv_logits = victim_model(adv_examples)
        adv_acc = calculate_accuracy(adv_logits, targets[: adv_examples.size(0)])
    predictions = adv_logits.argmax(dim=1)
    return predictions, adv_examples, float(adv_acc)


def run_attacks(trained_models: Dict[str, Dict[int, torch.nn.Module]], cfg: DictConfig, *, attack_repo: AttackRepository, writer: SummaryWriter) -> pd.DataFrame:
    """Run pairwise transfer attacks among the trained models."""

    epsilons = [i / 100 for i in range(0, 101, 2)]
    records, existing_pairs = attack_repo.existing_pairs()
    models_by_seed = _group_models_by_seed(trained_models)

    for seed, models in models_by_seed.items():
        log.info("\nRunning attacks for seed %s", seed)
        seed_experiment(seed)
        dataset, attack = _prepare_attack(cfg)
        targets = dataset.targets

        for attacker_name, attacker_model in models:
            log.info("Attacking with source model %s", attacker_name)
            attacker_model.eval()
            gradients_path = None
            compute_gradients = getattr(attack, "compute_gradients", None)
            if callable(compute_gradients):
                gradients = compute_gradients(attacker_model)
                gradients_path = attack_repo.save_gradients(attacker_model=attacker_name, seed=seed, gradients=gradients)

            attack_plan = [
                (victim_name, victim_model, epsilon)
                for victim_name, victim_model in models
                for epsilon in epsilons
                if (seed, attacker_name, victim_name) not in existing_pairs
            ]

            if not attack_plan:
                continue

            pbar = tqdm(attack_plan, desc="Attacking Models")
            for victim_name, victim_model, epsilon in pbar:
                pbar.set_postfix(info=victim_name)
                predictions, adv_examples, adv_acc = _evaluate_attack(
                    attack=attack,
                    attacker_model=attacker_model,
                    victim_model=victim_model,
                    epsilon=epsilon,
                    targets=targets,
                    device=cfg.device.device,
                )

                predictions_path = attack_repo.save_predictions(
                    target_model=victim_name,
                    seed=seed,
                    source_model=attacker_name,
                    epsilon=epsilon,
                    predictions=predictions,
                    targets=targets[: adv_examples.size(0)],
                )

                result = AttackResult(
                    seed=seed,
                    source_model=attacker_name,
                    target_model=victim_name,
                    epsilon=epsilon,
                    accuracy=adv_acc,
                    num_samples=adv_examples.size(0),
                    predictions_path=predictions_path,
                    gradients_path=gradients_path,
                )

                _log_tensorboard(writer, result)
                records.append(result.to_record())

    return pd.DataFrame(records)


@hydra.main(config_path="../conf", config_name="attack", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for running model transfer attacks."""

    run_dir = HydraConfig.get().run.dir
    logging_service = LoggingService(run_dir)
    results_dir = Path(cfg.results_path) if cfg.results_path else Path(logging_service.root_dir)
    if results_dir.suffix in {".csv", ".json"}:
        results_dir = results_dir.parent
    attack_repo = AttackRepository(results_dir)
    writer = SummaryWriter(log_dir=attack_repo.base_dir / "tensorboard")

    models = ModelRepository(cfg.models_path).load(load=cfg.model_load, map_location=cfg.device.device)
    profiler = attack_repo.profile(input_shape=(), epochs=0)
    with profiler.profile():
        attack_results = run_attacks(models, cfg, attack_repo=attack_repo, writer=writer)

    writer.flush()
    writer.close()

    attack_repo.save(attack_results)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")


if __name__ == "__main__":
    main()
