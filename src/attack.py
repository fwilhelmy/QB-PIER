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

from attacks import AttackFactory
from datasets import DatasetFactory
from loading import load_attacks, load_experiment
from LoggingService import LoggingService
from utils import calculate_accuracy, seed_experiment
from visualisation.attacks import plot_transfer_attack_effects

log = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Container for information gathered from a single attack run."""

    seed: int
    source_model: str
    target_model: str
    epsilon: float
    accuracy: float
    num_samples: int
    predictions_path: str | None
    gradients_path: str | None

    def to_record(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "epsilon": self.epsilon,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "predictions_path": self.predictions_path,
            "gradients_path": self.gradients_path,
        }


def _resolve_results_dir(cfg: DictConfig, logging_service: LoggingService) -> Path:
    """Return the directory where attack artefacts should be written."""

    if cfg.results_path:
        candidate = Path(cfg.results_path)
        if candidate.suffix in {".csv", ".json"}:
            return candidate.parent
        return candidate
    return Path(logging_service.root_dir)


def _results_csv_path(results_dir: Path) -> Path:
    return results_dir / "results.csv"


def _load_existing_results(cfg: DictConfig, results_dir: Path) -> Tuple[List[Dict[str, object]], set[Tuple[int, str, str]]]:
    """Load prior attack outcomes to avoid recomputing pairs."""

    records: List[Dict[str, object]] = []
    existing_pairs: set[Tuple[int, str, str]] = set()

    candidate_paths = []
    if cfg.results_path:
        path = Path(cfg.results_path)
        candidate_paths.append(path if path.suffix else path / "results.csv")
    candidate_paths.append(_results_csv_path(results_dir))

    for path in candidate_paths:
        if not path or not path.exists():
            continue
        try:
            previous_results = load_attacks(str(path))
            records.extend(previous_results.to_dict("records"))
            existing_pairs = set(
                zip(
                    previous_results["seed"],
                    previous_results["source_model"],
                    previous_results["target_model"],
                )
            )
            log.info("Loaded %s existing attack results from '%s'.", len(previous_results), path)
            break
        except Exception as ex:  # pragma: no cover - defensive path
            log.error("Problem loading existing results from '%s': %s", path, ex)

    return records, existing_pairs


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


def _save_predictions(
    *,
    base_dir: Path,
    target_model: str,
    seed: int,
    source_model: str,
    epsilon: float,
    predictions: Tensor,
    targets: Sequence[int],
) -> str:
    """Persist model predictions for inspection."""

    model_dir = base_dir / target_model
    model_dir.mkdir(parents=True, exist_ok=True)
    file_path = model_dir / "predictions.csv"

    df = pd.DataFrame(
        {
            "seed": seed,
            "source_model": source_model,
            "target_model": target_model,
            "epsilon": epsilon,
            "sample_index": range(len(predictions)),
            "target": targets,
            "prediction": predictions.cpu().tolist(),
        }
    )

    header = not file_path.exists()
    df.to_csv(file_path, mode="a", header=header, index=False)
    return str(file_path)


def _save_gradients(*, base_dir: Path, attacker_model: str, seed: int, gradients: List[Tuple[Tensor, Tensor]]) -> str:
    model_dir = base_dir / attacker_model
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"gradients_seed{seed}.pt"
    torch.save({"seed": seed, "model": attacker_model, "gradients": gradients}, path)
    return str(path)


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


def run_attacks(trained_models: Dict[str, Dict[int, torch.nn.Module]], cfg: DictConfig, *, results_dir: Path, writer: SummaryWriter) -> pd.DataFrame:
    """Run pairwise transfer attacks among the trained models."""

    epsilons = [i / 100 for i in range(0, 101, 2)]
    records, existing_pairs = _load_existing_results(cfg, results_dir)
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
                gradients_path = _save_gradients(
                    base_dir=results_dir,
                    attacker_model=attacker_name,
                    seed=seed,
                    gradients=gradients,
                )

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

                predictions_path = _save_predictions(
                    base_dir=results_dir,
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
    results_dir = _resolve_results_dir(cfg, logging_service)
    results_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=results_dir / "tensorboard")

    models = load_experiment(
        cfg.models_path,
        load=cfg.model_load,
        map_location=cfg.device.device,
    )

    attack_results = run_attacks(models, cfg, results_dir=results_dir, writer=writer)

    writer.flush()
    writer.close()

    results_csv = _results_csv_path(results_dir)
    results_json = results_dir / "results.json"
    attack_results.to_csv(results_csv, index=False)
    attack_results.to_json(results_json, orient="records", indent=2)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")


if __name__ == "__main__":
    main()
