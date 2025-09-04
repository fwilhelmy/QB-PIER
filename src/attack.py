from collections import defaultdict
import os

import pandas as pd
import torch
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from metrics import LoggingService
from utils import calculate_accuracy, seed_experiment
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks import AttackFactory
from loading import load_experiment
from visualisation.attacks import plot_transfer_attack_effects


def run_attacks(
    trained_models,
    dataset_type: str,
    device,
    attack_name: str,
    attack_params: dict,
    epsilons,
    dataset_root: str = "./data/processed",
):
    """Run pairwise transfer attacks among the trained models."""

    records = []

    models_by_seed = defaultdict(list)
    for model_seeds in trained_models.values():
        for seed, model in model_seeds.items():
            models_by_seed[seed].append(model)

    attack_args = {k: v for k, v in attack_params.items() if k != "name"}

    for seed, models in models_by_seed.items():
        print(f"\n>>> Running attacks for seed : {seed}")
        seed_experiment(seed)

        dm = DatasetManagerFactory.create(dataset_type, root=dataset_root)
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset
        targets = dataset.targets

        attack = AttackFactory.create(attack_name, dataset=dm.dataset, **attack_args)

        for attacker_model in models:
            print(f"Attacking with source model {attacker_model.model_path}")
            attacker_model.eval()
            pbar = tqdm([(model, epsilon) for model in models for epsilon in epsilons],
                        desc="Attacking Models")
            for victim_model, epsilon in pbar:
                pbar.set_postfix(info=victim_model.model_path)
                adv_examples = attack.attack(attacker_model, epsilon).to(device)
                victim_model.eval()
                with torch.no_grad():
                    adv_logits = victim_model(adv_examples)
                    adv_acc = calculate_accuracy(
                        adv_logits,
                        targets[: adv_examples.size(0)],
                    )
                records.append(
                    {
                        "seed": seed,
                        "source_model": attacker_model.model_path,
                        "target_model": victim_model.model_path,
                        "epsilon": epsilon,
                        "accuracy": float(adv_acc),
                    }
                )

    return pd.DataFrame(records)

@hydra.main(config_path="../conf", config_name="attack", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for running model transfer attacks."""
    run_dir = HydraConfig.get().run.dir
    ls = LoggingService(run_dir)
    models = load_experiment(
        cfg.experiment_dir, load=cfg.model_load, map_location=cfg.device.device
    )
    if 100 % cfg.stride != 0:
        raise ValueError("stride must divide 100")
    epsilons = [i / 100 for i in range(0, 101, cfg.stride)]
    attack_results = run_attacks(
        models,
        cfg.dataset.dataset,
        cfg.device.device,
        cfg.attack.name,
        cfg.attack,
        epsilons,
        cfg.dataset.root,
    )
    attack_path = os.path.join(ls.root_dir, "attacks")
    os.makedirs(attack_path, exist_ok=True)
    attack_results.to_csv(os.path.join(attack_path, "results.csv"), index=False)
    attack_results.to_json(os.path.join(attack_path, "results.json"), index=False)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")


if __name__ == "__main__":
    main()
