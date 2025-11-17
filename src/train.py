import logging
import os
import time
from typing import Optional

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig
from torchsummary import summary

from DiscordWebhookLogger import DiscordWebhookLogger
from datasets import DatasetFactory
from LoggingService import LoggingService
from models.ModelFactory import ModelFactory
from utils import seed_experiment
from visualisation.train import generate_gradient_flow_plots, plot_training_curves

log = logging.getLogger(__name__)


def train_model(
    cfg: DictConfig,
    log_dir: str,
    discord_logger: Optional[DiscordWebhookLogger] = None,
):
    """
    Runs a full training pipeline once per seed and returns all trained models.
    
    Args:
        cfg: Experiment configuration as a dictionary resolved from Hydra.
        log_dir: Relative logging directory inside the logging service root.
        discord_logger: Optional webhook helper for experiment notifications.

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

    discord_logger.log_exp_start(model_name, seeds)

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
        start_time = time.time()
        try:
            model.fit(train_loader, eval_loader)
            trained[seed] = model
            log.info("Training completed for model %s (seed=%s).", model_name, seed)
            elapsed = time.time() - start_time
            seed_times[seed] = elapsed
            if discord_logger:
                discord_logger.log_seed_completed(model_name, seed, elapsed)
        except Exception as e:
            log.warning("Training failed for model %s (seed=%s).", model_name, seed)
            elapsed = time.time() - start_time
            seed_times[seed] = elapsed
            if discord_logger:
                discord_logger.log_seed_failed(model_name, seed, str(e))

    if len(trained) != 0:
        avg_time = sum(seed_times.values()) / len(seed_times) if seed_times else 0.0
        model = list(trained.values())[0]
        info = str(summary(model, input_data=tuple(cfg["dataset"]["input_shape"]), col_names=("num_params","input_size","output_size"),verbose=0))
        epochs = cfg["training"].get("epochs", 0)
        info += f"\nTrained epochs: {epochs}"
        info += f"\nTraining time per epoch: {avg_time/epochs:.2f} seconds"
        info += f"\nTotal training time (per seed): {avg_time:.2f} seconds"
        LoggingService().save_file(info, "model.info", path=log_dir)
    return trained

@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run an experiment using a Hydra configuration."""
    root_dir = HydraConfig.get().run.dir.rsplit("/",1)[0]
    logging_service = LoggingService(root_dir=root_dir)

    log_dir = HydraConfig.get().job.override_dirname if HydraConfig.get().mode  == RunMode.MULTIRUN else cfg["model"]["model_class"]
    log.info(f"Starting experiment ({log_dir})")

    discord_logger = DiscordWebhookLogger()

    trained = train_model(cfg, log_dir, discord_logger)

    if not cfg["training"].get("fast_dev_run", False):
        plot_training_curves(cfg["model"]["model_class"], trained, log_dir)
        if trained:
            generate_gradient_flow_plots(list(trained.values())[0], log_dir)
        figure_path = os.path.join(
            logging_service.root_dir,
            log_dir,
            f"training_{cfg['model']['model_class']}.png",
        )
        if discord_logger and os.path.exists(figure_path):
            discord_logger.log_training_curves(cfg["model"]["model_class"], figure_path)
    logging_service.save_experiment()
    if discord_logger:
        seeds_cfg = cfg["training"].get("seeds", [0, 42]) or []
        discord_logger.log_experiment_finished(
            cfg["model"]["model_class"],
            trained=len(trained),
            expected=len(seeds_cfg),
            log_dir=os.path.join(logging_service.root_dir, log_dir),
        )

if __name__ == "__main__":
    main()

