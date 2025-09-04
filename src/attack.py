from collections import defaultdict
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

from metrics import LoggingService
from utils import calculate_accuracy, seed_experiment
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks import FGSMAttack
from loading import load_experiment
from visualisation.training import plot_training_curves, plot_models_together
from visualisation.attacks import plot_transfer_attack_effects


def run_attacks(trained_models, dataset_type: str, device):
    """Run pairwise transfer attacks among the trained models."""
    records = []

    models_by_seed = defaultdict(list)
    for model_seeds in trained_models.values():
        for seed, model in model_seeds.items():
            models_by_seed[seed].append(model)

    for seed, models in models_by_seed.items():
        print(f"\n>>> Running attacks for seed : {seed}")
        seed_experiment(seed)

        # Prepare dataset and gradients once for efficiency
        dm = DatasetManagerFactory(dataset_type)
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset
        data = dataset.data.unsqueeze(1).to(torch.float)
        targets = dataset.targets

        attack = FGSMAttack(dm.dataset)
        stride = 2
        epsilons = [x / 100.0 for x in range(0, 100 + stride, stride)]

        for attacker_model in models:
            print(f"Attacking with source model {attacker_model.model_path}")
            attacker_model.eval()
            attack.compute_gradients(attacker_model)
            pbar = tqdm([(model, epsilon) for model in models for epsilon in epsilons],
                        desc="Attacking Models")
            for victim_model, epsilon in pbar:
                pbar.set_postfix(info=victim_model.model_path)
                victim_model.eval()
                adv_examples = attack.attack(attacker_model, epsilon).to(device)
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


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python loading.py <root_dir>")
    #     sys.exit(1)
    # root_dir = sys.argv[1]
    root_dir = "logs/default/2025-09-03@17-16-07"
    ls = LoggingService("logs")
    models = load_experiment(root_dir, load="final", map_location="cpu")
    # for path, seeds in models.items():
    #     plot_training_curves("Model", seeds, path)
    # plot_models_together(models)
    attack_results = run_attacks(models, "mnist", "cpu")
    os.makedirs(os.path.join(ls.root_dir, "attacks"), exist_ok=True)
    attack_path = os.path.join(ls.root_dir, "attacks")
    attack_results.to_csv(os.path.join(attack_path, "results.csv"), index=False)
    attack_results.to_json(os.path.join(attack_path, "results.json"), index=False)
    plot_transfer_attack_effects(attack_results, title_prefix="attacks/")

