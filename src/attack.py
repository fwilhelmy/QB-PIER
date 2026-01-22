"""Tools for running transfer attacks between trained models.

The module organises attack execution, logging, and artefact persistence
into focused helper functions for easier maintenance.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
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
from attacks.AttackResult import AttackResult
from datasets import DatasetFactory
from LoggingService import LoggingService
from models.ModelFactory import ModelFactory
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


def _crawl_model_runs(base_dir: Path) -> List[Path]:
    experiments: List[Path] = []
    for root, _, files in os.walk(base_dir):
        if ".model" in files:
            experiments.append(Path(root))
    experiments.sort()
    return experiments


def _score_checkpoint(path: Path) -> float:
    match = re.search(r"loss=(\d+\.\d+)", path.name)
    return float(match.group(1)) if match else float("inf")


def _best_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    ckpts = sorted([f for f in checkpoint_dir.iterdir() if f.suffix == ".ckpt"], key=_score_checkpoint)
    return ckpts[0] if ckpts else None


def _select_checkpoint(seed_dir: Path, load: str) -> Path | None:
    final_ckpt = seed_dir / "final.ckpt"
    if load == "best":
        best_ckpt = _best_checkpoint(seed_dir / "checkpoints")
        return best_ckpt or (final_ckpt if final_ckpt.exists() else None)
    if final_ckpt.exists():
        return final_ckpt
    return _best_checkpoint(seed_dir / "checkpoints")


def load_trained_models(
    base_dir: str | Path,
    *,
    load: str = "final",
    map_location: str = "cpu",
) -> Dict[str, Dict[int, torch.nn.Module]]:
    experiments: Dict[str, Dict[int, torch.nn.Module]] = {}
    base_dir = Path(base_dir)
    for model_path in tqdm(_crawl_model_runs(base_dir), desc="Loading experiment"):
        model_name = (model_path / ".model").read_text().strip()
        experiment_name = os.path.relpath(model_path, base_dir)
        models: Dict[int, torch.nn.Module] = {}
        for seed_dir in model_path.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            checkpoint = _select_checkpoint(seed_dir, load)
            if checkpoint is None:
                log.warning("No checkpoint found for %s (seed=%s).", model_name, seed_dir.name)
                continue
            model = ModelFactory.load(
                model_name,
                checkpoint=str(checkpoint),
                map_location=map_location,
                log_dir=str(seed_dir),
            )
            models[int(seed_dir.name)] = model
        if models:
            experiments[experiment_name] = models
    return experiments


def _load_results(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _existing_pairs(results_dir: Path) -> Tuple[List[AttackResult], set[Tuple[int, str, str]]]:
    data = _load_results(results_dir)
    if data.empty:
        return [], set()
    pairs = set(zip(data["seed"], data["source_model"], data["target_model"]))
    results = [AttackResult.from_record(record) for record in data.to_dict("records")]
    return results, pairs


def _save_predictions(
    *,
    results_dir: Path,
    target_model: str,
    seed: int,
    source_model: str,
    epsilon: float,
    predictions: Tensor,
    targets: Sequence[int],
) -> str:
    model_dir = results_dir / target_model
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
    df.to_csv(file_path, mode="a", header=not file_path.exists(), index=False)
    return str(file_path)


def _save_gradients(
    *,
    results_dir: Path,
    attacker_model: str,
    seed: int,
    gradients: List[Tuple[Tensor, Tensor]],
) -> str:
    model_dir = results_dir / attacker_model
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"gradients_seed{seed}.pt"
    torch.save({"seed": seed, "model": attacker_model, "gradients": gradients}, path)
    return str(path)


def _save_attack(
    *,
    results_dir: Path,
    seed: int,
    source_model: str,
    target_model: str,
    epsilon: float,
    accuracy: float,
    predictions: Tensor,
    targets: Sequence[int],
    num_samples: int,
    gradients_path: str | None = None,
    gradients: List[Tuple[Tensor, Tensor]] | None = None,
) -> AttackResult:
    if gradients_path is None and gradients is not None:
        gradients_path = _save_gradients(
            results_dir=results_dir,
            attacker_model=source_model,
            seed=seed,
            gradients=gradients,
        )

    predictions_path = _save_predictions(
        results_dir=results_dir,
        target_model=target_model,
        seed=seed,
        source_model=source_model,
        epsilon=epsilon,
        predictions=predictions,
        targets=targets,
    )

    return AttackResult(
        seed=seed,
        source_model=source_model,
        target_model=target_model,
        epsilon=epsilon,
        accuracy=accuracy,
        num_samples=num_samples,
        predictions_path=predictions_path,
        gradients_path=gradients_path,
    )


def _save_results(results_dir: Path, results: Iterable[AttackResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_record() for r in results])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "results.csv", index=False)
    df.to_json(results_dir / "results.json", orient="records", indent=2)
    return df


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


def run_attacks(
    trained_models: Dict[str, Dict[int, torch.nn.Module]],
    cfg: DictConfig,
    *,
    results_dir: Path,
    writer: SummaryWriter,
) -> List[AttackResult]:
    """Run pairwise transfer attacks among the trained models."""

    epsilons = [i / 100 for i in range(0, 101, 2)]
    records, existing_pairs = _existing_pairs(results_dir)
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
                    results_dir=results_dir,
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

                result = _save_attack(
                    results_dir=results_dir,
                    seed=seed,
                    source_model=attacker_name,
                    target_model=victim_name,
                    epsilon=epsilon,
                    accuracy=adv_acc,
                    predictions=predictions,
                    targets=targets[: adv_examples.size(0)],
                    num_samples=adv_examples.size(0),
                    gradients_path=gradients_path,
                )

                _log_tensorboard(writer, result)
                records.append(result)

    return records


@hydra.main(config_path="../conf", config_name="attack", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for running model transfer attacks."""

    run_dir = HydraConfig.get().run.dir
    logging_service = LoggingService(run_dir)
    results_dir = Path(cfg.results_path) if cfg.results_path else Path(logging_service.root_dir)
    if results_dir.suffix in {".csv", ".json"}:
        results_dir = results_dir.parent
    writer = SummaryWriter(log_dir=results_dir / "tensorboard")

    models = load_trained_models(cfg.models_path, load=cfg.model_load, map_location=cfg.device.device)
    attack_results = run_attacks(models, cfg, results_dir=results_dir, writer=writer)

    writer.flush()
    writer.close()

    results_df = _save_results(results_dir, attack_results)
    plot_transfer_attack_effects(results_df, title_prefix="attacks/")


if __name__ == "__main__":
    main()
