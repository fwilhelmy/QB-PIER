"""Standalone script for running PyTorch Lightning's learning rate finder."""

import json
import logging
import os
from typing import Any, Dict

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from datasets import DatasetFactory
from metrics import LoggingService
from models.ModelFactory import ModelFactory
from utils import seed_experiment

log = logging.getLogger(__name__)


def _build_dataloaders(cfg: DictConfig):
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
        num_workers=cfg["device"]["num_workers"],
    )
    return train_loader, eval_loader


def _build_model(cfg: DictConfig, log_dir: str):
    model_name = cfg["model"]["model_class"].lower()
    model_config = {
        k: v
        for k, v in cfg["model"].items()
        if k not in {"model_class", "model_type"}
    }

    model = ModelFactory.create(
        model_name,
        model_path=log_dir,
        input_size=cfg["dataset"]["input_shape"],
        output_size=cfg["dataset"]["classes"],
        **model_config,
        configure={
            "optimizer": cfg["training"]["optimiser"]["name"],
            "learning_rate": cfg["training"]["optimiser"].get(
                "learning_rate", cfg["training"].get("learning_rate", 1e-3)
            ),
            "max_steps": cfg["training"]["n_steps"],
            "optimiser_config": cfg["training"]["optimiser"].get(
                "optimiser_config", {}
            ),
            "accelerator": cfg["device"]["accelerator"],
            "devices": cfg["device"]["devices"],
            "log_interval": cfg["training"]["log_interval"],
        },
    )
    return model.to(cfg["device"]["device"])


def _run_for_seed(
    cfg: DictConfig,
    seed: int,
    base_log_dir: str,
    lr_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    log.info("Running LR finder for seed=%s ...", seed)
    seed_experiment(seed)

    train_loader, eval_loader = _build_dataloaders(cfg)
    model = _build_model(cfg, os.path.join(base_log_dir, str(seed)))

    tuner = getattr(model.trainer, "tuner", None)
    if tuner is None:
        raise RuntimeError(
            "Trainer tuner is not available; ensure PyTorch Lightning >=1.5"
        )

    try:
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader,
            min_lr=lr_cfg.get("min_lr", 1e-6),
            max_lr=lr_cfg.get("max_lr", 1.0),
            num_training=lr_cfg.get("num_training", 100),
            mode=lr_cfg.get("mode", "exponential"),
            early_stop_threshold=lr_cfg.get("early_stop_threshold", 4.0),
        )
    except MisconfigurationException as exc:
        log.error("LR finder misconfiguration: %s", exc)
        return {"error": str(exc)}
    except Exception as exc:  # pragma: no cover - robustness
        log.exception("Unexpected error during LR finder")
        return {"error": str(exc)}

    suggestion = lr_finder.suggestion()
    result_dir = os.path.join(base_log_dir, str(seed))

    payload: Dict[str, Any] = {
        "suggestion": suggestion,
        "results": lr_finder.results,
        "search_space": {
            "min_lr": lr_cfg.get("min_lr", 1e-6),
            "max_lr": lr_cfg.get("max_lr", 1.0),
            "num_training": lr_cfg.get("num_training", 100),
            "mode": lr_cfg.get("mode", "exponential"),
            "early_stop_threshold": lr_cfg.get("early_stop_threshold", 4.0),
        },
    }

    LoggingService().save_file(
        json.dumps(payload, indent=2),
        "lr_finder.json",
        path=result_dir,
    )

    figure = lr_finder.plot(suggest=True)
    if figure is not None:
        LoggingService().save_figure(figure, "lr_finder", path=result_dir)

    reset_fn = getattr(tuner, "reset_lr_find", None)
    if callable(reset_fn):
        reset_fn()

    if suggestion is not None:
        log.info("Suggested learning rate for seed %s: %.2e", seed, suggestion)
    else:
        log.info("No learning rate suggestion found for seed %s.", seed)

    return payload


def run_lr_sweep(cfg: DictConfig, log_dir: str) -> Dict[int, Dict[str, Any]]:
    seeds = cfg["training"].get("seeds", [0])
    num_threads = cfg["device"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    LoggingService().save_file("lr_finder", ".mode", path=log_dir)
    LoggingService().save_file(
        cfg["model"]["model_class"].lower(),
        ".model",
        path=log_dir,
    )

    lr_cfg = dict(cfg.get("lr_finder", {}))
    results: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        results[seed] = _run_for_seed(cfg, seed, log_dir, lr_cfg)

    LoggingService().save_file(
        json.dumps(results, indent=2),
        "summary.json",
        path=log_dir,
    )
    return results


@hydra.main(config_path="../conf", config_name="lr_finder", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the standalone learning rate finder."""
    root_dir = HydraConfig.get().run.dir.rsplit("/", 1)[0]
    logging_service = LoggingService(root_dir=root_dir)

    log_dir = "lr_finder"
    subdir = HydraConfig.get().job.override_dirname
    if HydraConfig.get().mode == RunMode.MULTIRUN and subdir:
        log_dir = os.path.join(log_dir, subdir)

    log.info(
        "Exploring learning rates for '%s'",
        cfg["model"]["model_class"],
    )
    run_lr_sweep(cfg, log_dir)


if __name__ == "__main__":
    main()
