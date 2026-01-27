import logging
import os
import time
from datetime import datetime
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from DiscordWebhookLogger import DiscordWebhookLogger
from datasets import DatasetFactory
from LoggingService import LoggingService, ExperimentManager
from models.ModelFactory import ModelFactory

from utils import seed_experiment
from visualisation.train import generate_gradient_flow_plots, plot_training_curves

log = logging.getLogger(__name__)

def train_model(cfg: DictConfig, log_dir: str):
    """
    Runs a full training pipeline once per seed and returns all trained models.
    
    Args:
        cfg: Experiment configuration as a dictionary resolved from Hydra.
        log_dir: Relative logging directory inside the logging service root.

    Returns:
        dict: { seed: trained_model_instance }
    """
    trained_models = {}
    seeds = cfg["training"]["seeds"]
    
    model_name = cfg["model"]["model_class"].lower()
    model_config = {
        k: v for k, v in cfg["model"].items()
        if k not in {"model_class", "model_type"}
    }

    # Create an anchor file for later discovery
    root_dir = LoggingService().root_dir
    model_dir = os.path.join(root_dir, log_dir)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, ".model"), "w", encoding="utf-8") as file:
        file.write(model_name)

    num_threads = cfg["device"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    for seed in seeds:
        log.info(f"Starting training for seed %s", seed)

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
        model_dir = os.path.join(LoggingService().root_dir, log_dir, str(seed))
        model = ModelFactory.create(
            model_name,
            log_dir=model_dir,
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
            "lr_scheduler": cfg["training"].get("scheduler"),
            "fast_dev_run": cfg["training"].get("fast_dev_run", False),
        })

        # 4) Training
        if model.fit(train_loader, eval_loader):
            trained_models[seed] = model
            log.info("Training completed for model %s (seed=%s).", model_name, seed)
        else: log.exception("Training failed for model %s (seed=%s).", model_name, seed)

    return trained_models

@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run an experiment using a Hydra configuration."""
    root_dir = HydraConfig.get().run.dir.rsplit("/",1)[0]
    logging_service = LoggingService(root_dir=root_dir)
    fast_dev_run: bool = cfg["training"].get("fast_dev_run", False)

    model_name = cfg["model"]["model_class"]
    seeds = cfg["training"]["seeds"]
    log_dir = HydraConfig.get().job.override_dirname
    log.info(f"Starting experiment ({log_dir or model_name}) with seeds: {seeds}")

    discord_logger = DiscordWebhookLogger()

    if not fast_dev_run:
        discord_logger.log_exp_start(model_name, seeds, log_dir)
    experiment_dir = os.path.join(logging_service.root_dir, log_dir or "")
    with ExperimentManager(experiment_dir) as exp_info:
        trained_models = train_model(cfg, log_dir)

    if not fast_dev_run and trained_models:
        figure_path = plot_training_curves(model_name, trained_models, experiment_dir)
        generate_gradient_flow_plots(list(trained_models.values())[0], experiment_dir)
        discord_logger.log_exp_finished(
            model_name,
            trained=len(trained_models),
            expected=len(seeds),
            duration=exp_info["duration_seconds"],
            figure_path=figure_path,
            log_dir=experiment_dir,
        )

if __name__ == "__main__":
    main()
