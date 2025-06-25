import os
import random
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
    plot_training_result,
    plot_models_together,
    plot_training_curves,
    generate_gradient_flow_plots,
)
from arguments import Arguments
from models.classical.LeNet import LeNet
from models.classical.MLP import MLP
from models.classical.ResNet18 import ResNet18
from models.classical.MLPMixer import MLPMixer
from src.models.hybrid.HQuanvolution import HQuanvolution
from src.models.hybrid.HQPNN import HQPNN
from utils import seed_experiment, calculate_accuracy
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks.FGSMAttack import FGSMAttack
from attacks.PGDAttack import PGDAttack
from attacks.TransferAttack import run_transfer_attack
from LoggingService import LoggingService


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
            optimizer = args.optimizer,
            loss_fn = args.loss_fn,
            learning_rate = args.lr,
            max_steps = args.n_steps,
            weight_decay = args.weight_decay,
            momentum = args.momentum,
            devices = args.device,
            train_log_interval = args.train_log_interval,
            eval_log_interval = args.eval_log_interval
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
                df = run_transfer_attack(
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

def main():
    # Define a arguments and set global defaults
    LoggingService("newtests")
    seeds = [0]
    args = Arguments()
    args.verbose = False
    args.n_steps = 100
    args.num_workers = 0

    args.eval_ratio = 0.40
    args.train_ratio = 0.60

    args.train_batch_size = 2**8
    args.eval_batch_size = 2**11

    args.train_log_interval = 0.25
    args.eval_log_interval = 0.25

    # 2) Define your list of model‐configurations: (name, class, kwargs)
    model_specs = [
        # ("ResNet18_1", ResNet18, {
        #     "num_classes": 10
        # }),
        # ("MLPMixer_1", MLPMixer, {
        #     "num_classes": 10,
        #     "in_chans": 1,
        #     "img_size": 28,
        #     "patch_size": 4,
        #     "embed_dim": 256,
        #     "num_blocks": 4,
        #     "drop_rate": 0.0,
        #     "activation": "gelu",
        #     "mlp_ratio": [0.5, 4.0]
        # }),
        # ("MLP_1", MLP, {
        #     "input_size": (1, 28),
        #     "hidden_sizes": [1024, 512, 64, 64],
        #     "num_classes": 10,
        #     "activation": "relu"
        # }),
        ("HQNN_Parallel_1", HQPNN, {
            "in_channels": 1,
            "classes": 10
        }),
        ("LeNet_1", LeNet, {
            "in_channels": 1,
            "classes": 10
        }),
    ]

    # Train each model spec across all seeds, collect results as (model, seed)
    trained_models = {}
    for (model_name, model_cls, config) in model_specs:
        args.exp_name = model_name
        args.model = model_cls
        args.model_config = config

        trained_models[model_name] = train_model(args, seeds)

    # Run transfer attacks
    # attacks_metrics = run_attacks(trained_models, args)

    plot_training_curves(trained_models,
                     title_prefix="training")
    
    # plot_transfer_attack_effects(attacks_metrics,
    #                             title_prefix="transfer")
    # from pprint import pprint
    # pprint(trained_models['HQNN_Parallel_1'][0].all_metrics, depth=2)
    # metrics = trained_models['HQNN_Parallel_1'][0].all_metrics

    generate_gradient_flow_plots(trained_models)

if __name__ == "__main__":
    main()
