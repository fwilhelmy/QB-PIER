from collections import defaultdict
from math import e
import os

import pandas as pd
import torch
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from metrics import LoggingService
from utils import calculate_accuracy, seed_experiment
from datasets import DatasetFactory
from attacks import AttackFactory
from loading import load_attacks, load_experiment
from visualisation.attacks import plot_transfer_attack_effects


def run_attacks(trained_models, cfg: DictConfig):
    """Run pairwise transfer attacks among the trained models.

    If ``results_path`` is provided, existing results are loaded from the CSV file and
    reused so that previously computed (source model, target model) pairs are not
    recomputed.
    """

    if 100 % cfg.stride != 0:
        raise ValueError("stride must divide 100")
    epsilons = [i / 100 for i in range(0, 101, cfg.stride)]

    records = []
    existing_pairs = set()

    try:
        previous_results = load_attacks(cfg.results_path)
        records.extend(previous_results.to_dict("records"))
        existing_pairs = set(
            zip(
                previous_results["seed"],
                previous_results["source_model"],
                previous_results["target_model"],
                previous_results["epsilon"],
            )
        )
        if existing_pairs:
            print(
                f"Loaded {len(previous_results)} existing attack results from '{cfg.results_path}'."
            )
    except Exception as ex:
        formated_message = f"Problem loading existing results from '{cfg.results_path}': {ex}"
        LoggingService().log_error(formated_message)

    models_by_seed = defaultdict(list)
    for model_seeds in trained_models.values():
        for seed, model in model_seeds.items():
            models_by_seed[seed].append(model)

    attack_args = {k: v for k, v in cfg.attack.items() if k != "name"}

    for seed, models in models_by_seed.items():
        print(f"\n>>> Running attacks for seed : {seed}")
        seed_experiment(seed)

        dm = DatasetFactory.create(cfg.dataset.dataset, root=cfg.dataset.root)
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset
        targets = dataset.targets

        attack = AttackFactory.create(cfg.attack.name, dataset=dm.dataset, **attack_args)

        for attacker_model in models:
            print(f"Attacking with source model {attacker_model.model_path}")
            attacker_model.eval()
            pbar = tqdm([(model, epsilon) for model in models for epsilon in epsilons],
                        desc="Attacking Models")
            for victim_model, epsilon in pbar:
                pair_id = (
                    seed,
                    attacker_model.model_path,
                    victim_model.model_path,
                    epsilon,
                )
                if pair_id in existing_pairs:
                    print(f"Detected existing results for {pair_id}, skipping...")
                    continue
                pbar.set_postfix(info=victim_model.model_path)
                adv_examples = attack.attack(attacker_model, epsilon).to(cfg.device.device)
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
    ls = LoggingService(cfg.expdir)
    models = load_experiment(cfg.models_path, load=cfg.model_load, map_location=cfg.device.device
    )
    attack_results = run_attacks(models, cfg)
    attack_path = os.path.join(ls.root_dir, "attacks")
    os.makedirs(attack_path, exist_ok=True)
    attack_results.to_csv(os.path.join(attack_path, "results.csv"), index=False)
    attack_results.to_json(os.path.join(attack_path, "results.json"), index=False)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")

if __name__ == "__main__":
    main()