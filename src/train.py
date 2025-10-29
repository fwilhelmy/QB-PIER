from collections import defaultdict
import logging
import os
import time
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig
from torchsummary import summary

from datasets import DatasetFactory
from metrics import LoggingService
from models.ModelFactory import ModelFactory
from utils import seed_experiment
from visualisation.training import generate_gradient_flow_plots, plot_training_curves

log = logging.getLogger(__name__)


def train_model(cfg: DictConfig, log_dir: str):
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
    
    model_name = cfg["model"]["model_class"].lower()

    model_config = {
        k: v for k, v in cfg["model"].items()
        if k not in {"model_class", "model_type"}
    }

    # Create an anchor file for later discovery
    LoggingService().save_file(model_name, ".model", path=log_dir)

    num_threads = cfg["device"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    for idx, seed in enumerate(seeds, start=1):
        log.info(f"Starting training for seed {seed} ({idx}/{len(seeds)});")

        # 1) Seeding
        seed_experiment(seed)

        # 2) Prepare data
        dm = DatasetFactory.create(
            cfg["dataset"]["dataset"], root=cfg["dataset"].get("root")
        )
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
        scheduler_cfg = cfg["training"].get("scheduler")
        if isinstance(scheduler_cfg, DictConfig):
            scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)

        model = ModelFactory.create(
            model_name,
            model_path=os.path.join(log_dir, str(seed)),
            input_size=cfg["dataset"]["input_shape"],
            output_size=cfg["dataset"]["classes"],
            **model_config,
        )
        
        model.configure_training(config={
            "optimizer": cfg["training"]["optimiser"]["name"],
            "learning_rate": cfg["training"]["optimiser"].get("learning_rate", cfg["training"].get("learning_rate", 1e-3)),
            "max_epochs": cfg["training"]["epochs"],
            "optimiser_config": cfg["training"]["optimiser"].get("optimiser_config", {}),
            "accelerator": cfg["device"]["accelerator"],
            "devices": cfg["device"]["devices"],
            "lr_scheduler": scheduler_cfg,
            "fast_dev_run": cfg["training"].get("fast_dev_run", False),
        })

        # 4) Training
        start_time = time.time()
        model.fit(train_loader, eval_loader)
        elapsed = time.time() - start_time
        seed_times[seed] = elapsed
        trained[seed] = model

    avg_time = sum(seed_times.values()) / len(seed_times) if seed_times else 0.0
    if trained:
        # current_epoch is zero-indexed; add one to report completed epochs
        epochs_trained = getattr(model.trainer, "current_epoch", -1) + 1
        epochs = epochs_trained if epochs_trained > 0 else cfg["training"].get("epochs", 0)
    else:
        epochs = 0
    info = str(summary(model, input_data=tuple(cfg["dataset"]["input_shape"]), col_names=("num_params","input_size","output_size"),verbose=0))
    info += f"\nTrained epochs: {epochs}"
    if epochs:
        info += f"\nTraining time per epoch: {avg_time/epochs:.2f} seconds"
    else:
        info += "\nTraining time per epoch: N/A"
    info += f"\nTotal training time (per seed): {avg_time:.2f} seconds"
    LoggingService().save_file(info, "model.info", path=log_dir)

    return trained

@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run an experiment using a Hydra configuration."""
    root_dir = HydraConfig.get().run.dir.rsplit("/",1)[0]
    logging_service = LoggingService(root_dir=root_dir)

    model_name = log_dir = cfg["model"]["model_class"]
    subdir = HydraConfig.get().job.override_dirname
    if HydraConfig.get().mode  == RunMode.MULTIRUN and subdir:
        model_name += f" ({subdir})"
        log_dir = os.path.join(log_dir, subdir)
    log.info(f"Starting experiment '{model_name}'")

    trained = train_model(cfg, log_dir)
    plot_training_curves(cfg["model"]["model_class"], trained, log_dir)
    generate_gradient_flow_plots(list(trained.values())[0], log_dir)
    logging_service.save_experiment()

if __name__ == "__main__":
    main()

