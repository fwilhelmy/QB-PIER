import importlib
from collections import defaultdict
import os
from typing import List, Dict

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig

from visualisation.TrainingVisualisation import (
    plot_training_curves,
    generate_gradient_flow_plots,
)

import json
import time

from utils import seed_experiment, calculate_accuracy
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks import FGSMAttack
from metrics.LoggingService import LoggingService
from hydra.core.hydra_config import HydraConfig

def train_model(cfg: dict):
    """
    Runs a full training pipeline once per seed and returns all trained models.
    
    Args:
        cfg: Experiment configuration as a dictionary resolved from Hydra.
        seeds (list of int): Random seeds to use.

    Returns:
        dict: { seed: trained_model_instance }
    """
    trained = {}
    seed_times = {}

    seeds = cfg.get("seeds", [0, 42])
    
    module_path = f"models.{cfg['model']['model_type']}"
    model_module = importlib.import_module(module_path)
    model_class = getattr(model_module, cfg["model"]["model_class"])
    model_name = cfg["model"]["model_name"]
    model_dir = os.path.join(*HydraConfig.get().run.dir.rsplit("/", 2)[1:])
    model_config = cfg["model"]["params"] or {}

    print(f">>> Starting experiment '{model_name}' over seeds {seeds}")
    for idx, seed in enumerate(seeds, start=1):
        print(f"({idx}/{len(seeds)}); Seed = {seed}  ...")

        # Seeding (for reproducibility)
        seed_experiment(seed)

        num_threads = cfg["device"].get("num_threads")
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        # 2) Prepare data
        dm = DatasetManagerFactory(cfg["dataset"]["dataset"], root=cfg["dataset"].get("root"))
        train_split, eval_split = dm.calculate_split_sizes(
            cfg["dataset"]["train_ratio"], cfg["dataset"]["eval_ratio"]
        )
        train_loader, eval_loader = dm.split_and_load(
            train_samples_size=train_split,
            eval_samples_size=eval_split,
            train_batch_size=cfg["dataset"]["train_batch_size"],
            eval_batch_size=cfg["dataset"]["eval_batch_size"],
            num_workers=cfg["device"]["num_workers"]
        )

        # 3) Instantiate & configure the model
        model = model_class(
            model_name=model_name,
            seed=seed,
            input_size=cfg["dataset"]["input_shape"],
            output_size=cfg["dataset"]["classes"],
            **model_config
        )
        model.configure_training({
            "optimizer": cfg["training"]["optimizer"],
            "loss_fn": cfg["training"]["loss_fn"],
            "learning_rate": cfg["training"]["learning_rate"],
            "max_steps": cfg["n_steps"],
            "optimiser_config": cfg["training"].get("optimiser_config", {}),
            "accelerator": cfg["device"]["accelerator"],
            "devices": cfg["device"]["devices"],
            "train_log_interval": cfg.get("train_log_interval", 0.5),
            "eval_log_interval": cfg.get("eval_log_interval", 0.5),
        })
        model = model.to(cfg["device"]["device"])

        # 5) Training
        start_time = time.time()
        model.fit(train_loader, eval_loader)
        elapsed = time.time() - start_time
        seed_times[seed] = elapsed
        trained[seed] = model

    avg_time = sum(seed_times.values()) / len(seed_times) if seed_times else 0.0
    performance = {"seed_times": seed_times, "average": avg_time}
    LoggingService().save_file(
        content=json.dumps(performance, indent=2),
        title=f"performance.json",
        path=model_dir)

    return trained

def run_attacks(trained_models, cfg: dict):
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
        dm = DatasetManagerFactory(cfg["dataset"]["dataset"], root=cfg["dataset"].get("root"))
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset
        data = dataset.data.unsqueeze(1).to(torch.float)
        targets = dataset.targets

        attack = FGSMAttack(dm.dataset)
        epsilons = [x / 100.0 for x in range(0, 102, 2)]

        for attacker_model in models:
            attacker_model.eval()
            for victim_model in models:
                victim_model.eval()
                attack_results: List[Dict[str, float]] = []
                for eps in epsilons:
                    adv_examples = attack.attack(attacker_model, eps).to(cfg["device"]["device"])
                    with torch.no_grad():
                        adv_logits = victim_model(adv_examples)
                        adv_acc = calculate_accuracy(
                            adv_logits,
                            targets[: adv_examples.size(0)],
                        )
                    attack_results.append({"epsilon": eps, "accuracy": float(adv_acc)})

                for _, row in pd.DataFrame(attack_results).iterrows():
                    records.append(
                        {
                            "seed": seed,
                            "source_model": attacker_model.model_name,
                            "target_model": victim_model.model_name,
                            "epsilon": row["epsilon"],
                            "accuracy": row["accuracy"],
                        }
                    )

    return pd.DataFrame(records)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run an experiment using a Hydra configuration."""
    LoggingService()
    trained = train_model(cfg)
    plot_training_curves(cfg["model"]["model_name"], trained)
    generate_gradient_flow_plots(list(trained.values())[0])

if __name__ == "__main__":
    main()
