from collections import defaultdict
import importlib
import os
import time
import json
from typing import Dict, List

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch

from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig

from metrics import LoggingService
from utils import calculate_accuracy, seed_experiment
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks import FGSMAttack
from visualisation.training import plot_training_curves, generate_gradient_flow_plots


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

    seeds = cfg["training"].get("seeds", [0, 42])
    
    module_dir = f"models.{cfg['model']['model_type']}"
    model_module = importlib.import_module(module_dir)
    model_name = log_dir = cfg["model"]["model_class"]
    model_class = getattr(model_module, model_name)
    if HydraConfig.get().mode  == RunMode.MULTIRUN and HydraConfig.get().job.override_dirname:
        subdir = HydraConfig.get().job.override_dirname
        # if "." in subdir or "/" in subdir:
        #     subdir = subdir.split(".")[-1].split("/")[-1]
        model_name += f" ({subdir})"
        log_dir = os.path.join(log_dir, subdir)
    model_config = {
        k: v for k, v in cfg["model"].items()
        if k not in {"model_class", "model_type"}
    }

    # Create an anchor file for later discovery
    LoggingService().save_file("", ".model", path=log_dir)

    num_threads = cfg["device"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    print(f">>> Starting experiment '{model_name}' over seeds {seeds}")
    for idx, seed in enumerate(seeds, start=1):
        print(f"({idx}/{len(seeds)}); Seed = {seed}  ...")

        # 1) Seeding
        seed_experiment(seed)

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
            model_path=os.path.join(log_dir, str(seed)),
            input_size=cfg["dataset"]["input_shape"],
            output_size=cfg["dataset"]["classes"],
            **model_config
        )
        model.configure_training({
            "optimizer": cfg["training"]["optimiser"]["name"],
            "loss_fn": cfg["training"]["loss_fn"],
            "learning_rate": cfg["training"]["learning_rate"],
            "max_steps": cfg["training"]["n_steps"],
            "optimiser_config": cfg["training"]["optimiser"].get("optimiser_config", {}),
            "accelerator": cfg["device"]["accelerator"],
            "devices": cfg["device"]["devices"],
            "log_interval": cfg["training"]["log_interval"],
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
        path=log_dir)

    return log_dir, trained

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
    root_dir = HydraConfig.get().run.dir.rsplit("/",1)[0]
    LoggingService(root_dir=root_dir)
    log_dir, trained = train_model(cfg)
    plot_training_curves(cfg["model"]["model_class"], trained, log_dir)
    generate_gradient_flow_plots(list(trained.values())[0], log_dir)

if __name__ == "__main__":
    main()

