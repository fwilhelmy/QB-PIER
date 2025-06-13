import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd

from visualisation.AttacksVisualisation import (
    plot_attack_effect_on_accuracy,
    plot_training_curves,
    plot_transfer_attack_effects
)
from visualisation.TrainingVisualisation import (
    plot_training_result,
    plot_models_together,
)
from arguments import Arguments
from models.classical.LeNet import LeNet
from models.classical.MLP import MLP
from models.classical.ResNet18 import ResNet18
from models.classical.MLPMixer import MLPMixer
from models.hybrid.HQNN_Quanv import HQNN_Quanv
from models.hybrid.HQNN_Parallel import HQNN_Parallel
from utils import seed_experiment, calculate_accuracy
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks.FGSMAttack import FGSMAttack
from attacks.PGDAttack import PGDAttack
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
    """
    For each seed‐group in `trained_models`:
      - Re‐seed once (for repeatability within that seed)
      - Load test dataset
      - Instantiate FGSM/PGD attacks
      - Run transfer attacks only among models sharing the same seed
      - Print clean accuracy before and after generating adversarial examples
      - Plot & visualize results
    """
    records = []

    # Group models by seed
    models_by_seed = defaultdict(list)
    for model_seeds in trained_models.values():
        for seed in model_seeds.keys():
            models_by_seed[seed].append(model_seeds[seed])

    # For each seed group, run attacks among that seed’s models
    for seed, models in models_by_seed.items():
        print(f"\n>>> Running attacks for seed : {seed}")
        seed_experiment(seed)

        # Load the test dataset
        dataset_manager = DatasetManagerFactory(args.dataset)
        test_dataset = dataset_manager.load_dataset(train=False)
        test_data = test_dataset.data.unsqueeze(1).to(torch.float).to(args.device)
        test_targets = test_dataset.targets

        # Instantiate attack
        fgsm = FGSMAttack(test_dataset)
        epsilons = [x / 100.0 for x in range(0, 105, 5)]

        # For every "attacker" model in this seed group
        for attacker_model in models:
            attacker_name = attacker_model.model_name
            print(f"\n--- Attacking with model '{attacker_name}' (seed {seed}) ---")
            attacker_model.eval()

            # Compute clean (white‐box) accuracy BEFORE generating adversarial examples
            with torch.no_grad():
                logits_before = attacker_model(test_data)
                clean_acc_before = calculate_accuracy(logits_before, test_targets)
            print(f"Clean accuracy before generating adversarial examples: {clean_acc_before:.4f}")

            # For each ε, generate FGSM examples on attacker_model
            for ε in epsilons:
                adv_examples = fgsm.attack(attacker_model, ε).to(args.device)

                # Evaluate every victim on those adversarial examples
                with torch.no_grad():
                    for victim_model in models:
                        victim_name = victim_model.model_name
                        victim_model.eval()
                        adv_logits = victim_model(adv_examples)
                        adv_acc = calculate_accuracy(
                            adv_logits,
                            test_targets[:adv_examples.size(0)]
                        )
                        records.append({
                            "seed": seed,
                            "source_model": attacker_name,
                            "target_model": victim_name,
                            "epsilon": ε,
                            "accuracy": float(adv_acc)
                        })

            # Compute clean accuracy AGAIN to verify weights unchanged
            with torch.no_grad():
                logits_after = attacker_model(test_data)
                clean_acc_after = calculate_accuracy(logits_after, test_targets)
            print(f"Clean accuracy after generating adversarial examples: {clean_acc_after:.4f}")

    return pd.DataFrame(records)

def main():
    # Define a arguments and set global defaults
    LoggingService("newtests")
    seeds = [0]
    args = Arguments()
    args.verbose = False
    args.n_steps = 100
    args.num_workers = 0
    args.train_batch_size = 512
    args.eval_batch_size = 512

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
        # ("HQNN_Parallel_1", HQNN_Parallel, {
        #     "in_channels": 1,
        #     "classes": 10
        # }),
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
    from pprint import pprint
    pprint(trained_models['HQNN_Parallel_1'][0].all_metrics, depth=2)
    metrics = trained_models['HQNN_Parallel_1'][0].all_metrics

    print("Training metrics sizes:")
    print("loss:", len(metrics['train']['loss']))
    print("acc:", len(metrics['train']['acc']))
    print("step:", len(metrics['train']['step']))
    print("epoch:", len(metrics['train']['epoch']))

    print("\nValidation metrics sizes:")
    print("loss:", len(metrics['val']['loss']))
    print("acc:", len(metrics['val']['acc']))
    print("step:", len(metrics['val']['step']))
    print("epoch:", len(metrics['val']['epoch']))

if __name__ == "__main__":
    main()
