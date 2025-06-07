import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict

from visualisation.AttacksVisualisation import (
    plot_attack_effect_on_accuracy,
    visualize_attacks_results
)
from visualisation.TrainingVisualisation import (
    plot_training_result,
    plot_models_together
)
from arguments import Arguments
from models.classical.LeNet import LeNet
from models.classical.mlp import MLP
from models.classical.resnet18 import ResNet18
from models.classical.mlpmixer import MLPMixer
from models.hybrid.HQNN_Quanv import HQNN_Quanv
from models.hybrid.HQNN_Parallel import HQNN_Parallel
from utils import seed_experiment, calculate_accuracy
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks.FGSMAttack import FGSMAttack
from attacks.PGDAttack import PGDAttack


def train_model(args):
    """
    - Seed the experiment
    - Build data loaders
    - Instantiate & configure one model
    - Train it
    Returns the trained model instance (no run_path).
    """
    # 1) Seed for reproducibility
    seed_experiment(args.seed)

    # 2) Prepare data loaders
    dataset_manager = DatasetManagerFactory(args.dataset)
    train_split, eval_split = dataset_manager.calculate_split_sizes(
        args.train_ratio, args.eval_ratio
    )
    train_loader, eval_loader = dataset_manager.split_and_load(
        train_samples_size    = train_split,
        eval_samples_size     = eval_split,
        train_batch_size      = args.train_batch_size,
        eval_batch_size       = args.eval_batch_size,
        num_workers           = args.num_workers
    )

    # 3) Instantiate & configure the model
    model = args.model(
        model_name = args.exp_name,
        seed       = args.seed,
        **args.model_config
    )
    model.configure_training(
        optimizer      = args.optimizer,
        loss_fn        = args.loss_fn,
        learning_rate  = args.lr,
        max_steps      = args.n_steps,
        weight_decay   = args.weight_decay,
        momentum       = args.momentum,
        devices        = args.device,
    )
    model = model.to(args.device)

    # 4) (Optional) Print configuration if verbose
    if args.verbose:
        print("=" * 40)
        for k, v in vars(args).items():
            print(f"{k:20s}: {v}")
        print(f" rootdir      : {model.root_dir}")
        print(f" dataset_size : {len(dataset_manager.dataset)}")
        print("=" * 40)

    # 5) Train & plot training curves
    model.fit(train_loader, eval_loader)
    plot_training_result(model, train_split, eval_split)

    # 6) Return the trained model instance
    return model


def run_experiment(args, seeds=None):
    """
    Runs `train_model` multiple times (once per seed). 
    Returns a list of tuples: (trained_model_instance, seed)
    """
    if seeds is None:
        seeds = [0, 42]

    print(f">>> Starting experiment '{args.exp_name}' over seeds {seeds}")
    trained_models = []

    for idx, seed in enumerate(seeds):
        print(f"  ({idx+1}/{len(seeds)})  Seed = {seed}  ...")
        args.seed = seed

        trained_model = train_model(args)
        trained_models.append((trained_model, seed))

    return trained_models


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
    # 1) Group models by seed
    models_by_seed = defaultdict(list)
    for (model, seed) in trained_models:
        models_by_seed[seed].append(model)

    # 2) For each seed group, run attacks among that seed’s models
    for seed, models_list in models_by_seed.items():
        print(f"\n>>> Running attacks for seed = {seed}")

        # 2a) Re‐seed so that attack generation is deterministic within this group
        seed_experiment(seed)

        # 2b) Load the test dataset once for this seed
        dataset_manager = DatasetManagerFactory(args.dataset)
        test_dataset = dataset_manager.load_dataset(train=False)
        test_data    = test_dataset.data.unsqueeze(1).to(torch.float).to(args.device)
        test_targets = test_dataset.targets

        # 2c) Instantiate attack objects (they only need the raw test dataset)
        fgsm = FGSMAttack(test_dataset)
        pgd  = PGDAttack(test_dataset)

        # 2d) Choose which epsilons you want to sweep over
        epsilons = [x / 100.0 for x in range(0, 105, 5)]

        # 2e) For every “attacker” model in this seed group
        for attacker_model in models_list:
            attacker_model.eval()
            attacker_label = attacker_model.model_name
            print(f"\n--- Attacking with model '{attacker_label}' (seed {seed}) ---")

            # 2e-i) Compute clean (white‐box) accuracy BEFORE generating adversarial examples
            with torch.no_grad():
                logits_before = attacker_model(test_data)
                clean_acc_before = calculate_accuracy(logits_before, test_targets)
            print(f"Clean accuracy before generating adversarial examples: {clean_acc_before:.4f}")

            # 2e-ii) Prepare a container to hold “transfer accuracy” per victim
            transfer_results = {
                victim_model.model_name: {ε: None for ε in epsilons}
                for victim_model in models_list
            }

            # 2e-iii) For each ε, generate FGSM examples on attacker_model
            for ε in epsilons:
                adv_examples = fgsm.attack(attacker_model, ε).to(args.device)

                # Evaluate every victim on those adversarial examples
                with torch.no_grad():
                    for victim_model in models_list:
                        victim_model.eval()
                        adv_logits = victim_model(adv_examples)
                        adv_acc = calculate_accuracy(
                            adv_logits,
                            test_targets[:adv_examples.size(0)]
                        )
                        transfer_results[victim_model.model_name][ε] = adv_acc

                print(f"  ε={ε:.2f} → transfer acc per victim:")
                for v_label, eps_map in transfer_results.items():
                    print(f"     {v_label}: {eps_map[ε]:.4f}")

            # 2e-iv) Compute clean accuracy AGAIN to verify weights unchanged
            with torch.no_grad():
                logits_after = attacker_model(test_data)
                clean_acc_after = calculate_accuracy(logits_after, test_targets)
            print(f"Clean accuracy after generating adversarial examples: {clean_acc_after:.4f}")

            # 2e-v) Visualize & plot transfer curves for this “attacker”
            visualize_attacks_results(
                attacker_model,
                transfer_results
            )
            plot_attack_effect_on_accuracy(
                f"transfer_{attacker_label}_seed{seed}",
                [m for m in models_list],
                epsilons,
                {v_label: transfer_results[v_label] for v_label in transfer_results}
            )

        # 2f) (Optional) Run PGD similarly if desired:
        #     Just replace fgsm.attack(...) with pgd.attack(...) 


def main():
    # 1) Define a default Arguments() object and set global defaults
    seeds = [0, 42]
    args = Arguments()
    args.verbose           = False
    args.n_steps           = 100
    args.num_workers       = 0
    args.train_batch_size  = 512
    args.eval_batch_size   = 512
    # (Set your args.dataset, args.optimizer, args.loss_fn, args.lr, etc. here.)

    # 2) Define your list of model‐configurations: (name, class, kwargs)
    model_specs = [
        ("HQNN_Parallel_1", HQNN_Parallel, {"in_channels": 1, "classes": 10}),
        ("LeNet_1",         LeNet,        {"in_channels": 1, "classes": 10}),
        # … add more if needed …
    ]

    # 3) Train each model spec across all seeds, collect results as (model, seed)
    all_trained_models = []
    for (model_name, model_cls, config) in model_specs:
        args.exp_name     = model_name
        args.model        = model_cls
        args.model_config = config

        trained_models_for_this_spec = run_experiment(args, seeds)
        # Now each element is (model_instance, seed)
        all_trained_models.extend(trained_models_for_this_spec)

    # 4) Now run “within‐seed” transfer attacks
    run_attacks(all_trained_models, args)


if __name__ == "__main__":
    main()
