from collections import defaultdict
import logging
from math import e
import os

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sympy import root
from tqdm import tqdm

from attacks import AttackFactory
from datasets import DatasetFactory
from loading import load_attacks, load_experiment
from metrics import LoggingService
from utils import calculate_accuracy, seed_experiment
from visualisation.attacks import plot_transfer_attack_effects

log = logging.getLogger(__name__)


def run_attacks(trained_models, cfg: DictConfig):
    """Run pairwise transfer attacks among the trained models.

    If ``results_path`` is provided, existing results are loaded from the CSV file and
    reused so that previously computed (source model, target model) pairs are not
    recomputed.
    """

    epsilons = [i / 100 for i in range(0, 101, 2)]

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
            )
        )
        if existing_pairs:
            log.info(
                "Loaded %s existing attack results from '%s'.",
                len(previous_results),
                cfg.results_path,
            )
    except Exception as ex:
        formated_message = f"Problem loading existing results from '{cfg.results_path}': {ex}"
        log.error(formated_message)

    models_by_seed = defaultdict(list)
    for model_name, model_seeds in trained_models.items():
        for seed, model in model_seeds.items():
            models_by_seed[seed].append((model_name, model))

    attack_args = {k: v for k, v in cfg.attack.items() if k != "name"}

    for seed, models in models_by_seed.items():
        log.info("\n>>> Running attacks for seed : %s", seed)
        seed_experiment(seed)

        dm = DatasetFactory.create(cfg.dataset.dataset, root=cfg.dataset.root)
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset
        targets = dataset.targets

        attack = AttackFactory.create(cfg.attack.name, dataset=dm.dataset, **attack_args)

        for attacker_name, attacker_model in models:
            log.info("Attacking with source model %s", attacker_name)
            attacker_model.eval()
            attack_plan = []
            for victim_name, victim_model in models:
                pair_id = (
                    seed,
                    attacker_name,
                    victim_name,
                )
                if pair_id in existing_pairs:
                    log.info("Detected existing results for %s, skipping...", pair_id)
                    continue
                attack_plan.extend(
                    (victim_name, victim_model, epsilon) for epsilon in epsilons
                )

            if not attack_plan:
                continue

            pbar = tqdm(attack_plan, desc="Attacking Models")
            for victim_name, victim_model, epsilon in pbar:
                pbar.set_postfix(info=victim_name)
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
                        "source_model": attacker_name,
                        "target_model": victim_name,
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
    models = load_experiment(cfg.models_path, load=cfg.model_load, map_location=cfg.device.device)
    attack_results = run_attacks(models, cfg)
    os.makedirs(ls.root_dir, exist_ok=True)
    attack_results.to_csv(os.path.join(ls.root_dir, "results.csv"), index=False)
    attack_results.to_json(os.path.join(ls.root_dir, "results.json"), index=False)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")

if __name__ == "__main__":
    main()