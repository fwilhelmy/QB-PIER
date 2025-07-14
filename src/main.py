import importlib
from collections import defaultdict
from typing import List, Dict, Any

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

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


def train_model(cfg: dict, seeds=[0, 42]):
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

    model_name = cfg["model"]["model_name"]
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
        model_cls = cfg["model"]["class_obj"]
        model = model_cls(
            model_name=model_name,
            seed=seed,
            **cfg["model"]["params"]
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

        # 4) Print configuration
        if cfg.get("verbose", False):
            print("=" * 40)
            print(OmegaConf.to_yaml(OmegaConf.create(cfg)))
            print(f" rootdir      : {model.log_dir}")
            print(f" train_size: {len(train_loader.dataset)}")
            print(f" eval_size: {len(eval_loader.dataset)}")
            print(f"accelerator: {cfg['device']['accelerator']}")
            print(f"devices: {cfg['device']['devices']}")
            print(f"num_workers: {cfg['device']['num_workers']}")
            print(f"num_threads: {cfg['device'].get('num_threads', 'N/A')}")
            print(f"device: {cfg['device']['device']}")
            print("=" * 40)

        # 5) Training
        start_time = time.time()
        model.fit(train_loader, eval_loader)
        elapsed = time.time() - start_time
        seed_times[seed] = elapsed
        trained[seed] = model

    first_model = trained[seeds[0]]
    model_dir = f"{first_model.__class__.__name__}/{str(first_model.model_name)}"

    avg_time = sum(seed_times.values()) / len(seed_times) if seed_times else 0.0
    performance = {"seed_times": seed_times, "average": avg_time}
    LoggingService().save_file(
        content=json.dumps(performance, indent=2),
        title=f"performance.json",
        log_dir=model_dir)
    
    summary: Dict[str, Any] = {k: str(v) for k, v in cfg.items() if not k.startswith("_")}
    summary['seeds'] = seeds
    total = 0
    layers_params = {}
    for name, param in first_model.named_parameters():
        n = param.numel()
        layers_params[name] = n
        total += n
    layers_params['total'] = total
    summary['layers_params'] = layers_params
    summary['dataset'] = {
        "name": cfg["dataset"]["dataset"],
        "dataset_size": len(dm.dataset),
        "train_size": len(train_loader.dataset),
        "eval_size": len(eval_loader.dataset),
        "train_ratio": cfg["dataset"]["train_ratio"],
        "eval_ratio": cfg["dataset"]["eval_ratio"],
        "train_batch_size": cfg["dataset"]["train_batch_size"],
        "eval_batch_size": cfg["dataset"]["eval_batch_size"],
    }
    LoggingService().save_file(
        content=json.dumps(summary, indent=2),
        title=f"summary.json",
        log_dir=model_dir)

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
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Load model class using the new Hydra configuration fields
    model_cfg = cfg["model"]
    module_path = f"models.{model_cfg['model_type']}"
    model_module = importlib.import_module(module_path)
    model_cfg["class_obj"] = getattr(model_module, model_cfg["model_class"])

    # Resolve any dotted path parameters
    for key, val in list(model_cfg["params"].items()):
        if isinstance(val, str) and "." in val:
            m_path, c_name = val.rsplit(".", 1)
            mod = importlib.import_module(m_path)
            model_cfg["params"][key] = getattr(mod, c_name)

    cfg["model"] = model_cfg

    seeds = cfg.get("seeds", [0, 42])

    LoggingService(cfg["experiment_name"], logs_dir=cfg.get("LOGS_DIR"))
    trained = train_model(cfg, seeds)
    plot_training_curves(cfg["model"]["model_name"], trained)
    generate_gradient_flow_plots(list(trained.values())[0])

if __name__ == "__main__":    main()
