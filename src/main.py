import os
import random
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd

from visualisation.AttacksVisualisation import (
    plot_attack_effect_on_accuracy,
    plot_transfer_attack_effects,
)
from visualisation.TrainingVisualisation import (
    plot_models_together,
    plot_training_curves,
    generate_gradient_flow_plots,
)
from arguments import Arguments
from models.classical import (
    ResNet18,
    MLPMixer,
    MLP,
    ConvNet,
    LeNet,
)
from models.hybrid import (
    HQuanv,
    HQNN,
    HQE,
    DimVQC,
    AmpVQC,
)

from utils import seed_experiment, calculate_accuracy
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks import PGDAttack, FGSMAttack
from attacks.TransferAttack import transfer_attack
from metrics.LoggingService import LoggingService


def train_model(args, seeds=[0, 42]):
    """
    Runs a full training pipeline once per seed and returns all trained models.
    
    Args:
        args: An object with the experiment configuration (dataset, model, optimizer, etc.).
        seeds (list of int): Random seeds to use.

    Returns:
        dict: { seed: trained_model_instance }
    """
    trained = {}
    print(f">>> Starting experiment '{args.exp_name}' over seeds {seeds}")

    for idx, seed in enumerate(seeds, start=1):
        print(f"({idx}/{len(seeds)}); Seed = {seed}  ...")

        # Seeding (for reproducibility)
        args.seed = seed
        seed_experiment(seed)

        # 2) Prepare data
        dm = DatasetManagerFactory(args.dataset)
        train_split, eval_split = dm.calculate_split_sizes(
            args.train_ratio, args.eval_ratio
        )
        train_loader, eval_loader = dm.split_and_load(
            train_samples_size = train_split,
            eval_samples_size = eval_split,
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
            num_workers = args.num_workers
        )

        # 3) Instantiate & configure the model
        model = args.model(
            model_name = args.exp_name,
            seed = seed,
            **args.model_config
        )
        model.configure_training(
            {
                "optimizer": args.optimizer,
                "loss_fn": args.loss_fn,
                "learning_rate": args.lr,
                "max_steps": args.n_steps,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "accelerator": args.accelerator,
                "devices": args.devices,
                "train_log_interval": args.train_log_interval,
                "eval_log_interval": args.eval_log_interval,
            }
        )
        model = model.to(args.device)

        # 4) Print configuration
        if getattr(args, "verbose", False):
            print("=" * 40)
            for k, v in vars(args).items():
                print(f"{k:20s}: {v}")
            print(f" rootdir      : {model.root_dir}")
            print(f" dataset_size : {len(dm.dataset)}")
            print("=" * 40)

        # 5) Training
        model.fit(train_loader, eval_loader)
        trained[seed] = model

    return trained

def run_attacks(trained_models, args):
    """Run pairwise transfer attacks among the trained models."""
    records = []

    models_by_seed = defaultdict(list)
    for model_seeds in trained_models.values():
        for seed, model in model_seeds.items():
            models_by_seed[seed].append(model)

    for seed, models in models_by_seed.items():
        print(f"\n>>> Running attacks for seed : {seed}")
        seed_experiment(seed)

        for attacker_model in models:
            for victim_model in models:
                df = transfer_attack(
                    attacker_model,
                    victim_model,
                    dataset_name=args.dataset,
                    device=args.device,
                    epsilons=[x / 100.0 for x in range(0, 105, 5)],
                )
                for _, row in df.iterrows():
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

def main(args: Arguments | None = None):
    """Run the experiment with the provided configuration.

    Parameters
    ----------
    args : Arguments | None
        Configuration object. When ``None`` a new :class:`Arguments` instance
        with default values is created.
    """
    # Define arguments and set global defaults
    LoggingService("test_models")
    seeds = [0, 42]
    args = args or Arguments()
    args.verbose = True
    args.n_steps = 2000
    args.num_workers = 0
    
    if getattr(args, "num_threads", None) is not None:
        torch.set_num_threads(args.num_threads)

    args.eval_ratio = 0.4
    args.train_ratio = 0.6

    args.train_batch_size = 2**8
    args.eval_batch_size = 2**11

    args.train_log_interval = 0.5
    args.eval_log_interval = 0.5

    # 2) Define your list of model‐configurations: (name, class, kwargs)
    model_specs = [
        # ("ResNet18_1", ResNet18, {
        #     "input_size": (1, 28),
        #     "num_classes": 10,
        # }),
        # ("MLPMixer_1", MLPMixer, {
        #     "num_classes": 10,
        #     "input_size": (1, 28),
        #     "patch_size": 4,
        #     "embed_dim": 256,
        #     "num_blocks": 4,
        #     "drop_rate": 0.0,
        #     "activation": "gelu",
        #     "mlp_ratio": (0.5, 4.0),
        # }),
        # ("MLP_1", MLP, {
        #     "input_size": (1, 28),
        #     "hidden_sizes": [1024, 512, 64, 64],
        #     "num_classes": 10,
        #     "activation": "relu",
        # }),
        # ("ConvNet_1", ConvNet, {
        #     "input_size": (1, 28),
        #     "num_classes": 10,
        # }),
        # ("LeNet_1", LeNet, {
        #     "in_channels": 1,
        #     "classes": 10,
        # }),
        # ("HQNN_Parallel_1", HQNN, {
        #     "in_channels": 1,
        #     "classes": 10,
        # }),
        # ("HQuanv_1", HQuanv, {
        #     "in_channels": 1,
        #     "classes": 10,
        #     "img_size": 28,
        # }),
        # ("HQE_1", HQE, {
        #     "input_size": 28 * 28,
        #     "hidden_sizes": [256],
        #     "classes": 10,
        # }),
        # ("DimVQC_1", DimVQC, {
        #     "input_size": 28 * 28,
        #     "reduction_method": "pca",
        #     "reduced_dim": 4,
        #     "n_qdepth": 1,
        # }),
        ("AmpVQC_depth1", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 1,
        }),
        ("AmpVQC_depth2", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 2,
        }),
        ("AmpVQC_depth3", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 3,
        }),
        ("AmpVQC_depth4", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 4,
        }),
    ]

    # Train each model spec across all seeds, collect results as (model, seed)
    trained_models = {}
    for (model_name, model_cls, config) in model_specs:
        args.exp_name = model_name
        args.model = model_cls
        args.model_config = config

        trained_models[model_name] = train_model(args, seeds)
        plot_training_curves(model_name, trained_models[model_name])
        generate_gradient_flow_plots(list(trained_models[model_name].values())[0])

    plot_models_together(trained_models)  

    # Run transfer attacks
    # attacks_metrics = run_attacks(trained_models, args)
    # plot_transfer_attack_effects(attacks_metrics,
    #                             title_prefix="transfer")
    # from pprint import pprint
    # pprint(trained_models['HQNN_Parallel_1'][0].all_metrics, depth=2)
    # metrics = trained_models['HQNN_Parallel_1'][0].all_metrics

def _parse_cli() -> Arguments:
    """Parse a minimal set of command line options for experimentation."""
    parser = argparse.ArgumentParser(description="Run QB-PIER main entry point")
    parser.add_argument("--device", type=str, help="torch device, e.g. 'cpu' or 'cuda'")
    parser.add_argument("--accelerator", type=str, help="accelerator backend for lightning")
    parser.add_argument("--devices", type=int, help="number of devices to use")
    parser.add_argument("--num_workers", type=int, help="dataloader worker count")
    parser.add_argument("--num_threads", type=int, help="override torch num threads")

    cli_args = parser.parse_args()

    args = Arguments()
    for field in ["device", "accelerator", "devices", "num_workers", "num_threads"]:
        value = getattr(cli_args, field)
        if value is not None:
            setattr(args, field, value)
    return args


if __name__ == "__main__":
    cli_args = _parse_cli()
    main(cli_args)
