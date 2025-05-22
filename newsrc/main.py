import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import math
from visualisation.AttacksVisualisation import plot_attack_effect_on_accuracy, visualize_attacks_results
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together

from train import train
from utils import seed_experiment
from data.DatasetManagerFactory import DatasetManagerFactory

import torch

def train_model(model, args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)
    
    # Create a directory to save the experiment results
    checkpoint_path = os.path.join(args.log_dir, str(args.exp_name), str(args.exp_id))
    os.makedirs(checkpoint_path, exist_ok=True)

    # Data
    dataset_manager = DatasetManagerFactory(args.dataset)
    train_split, eval_split = dataset_manager.calculate_split_sizes(args.train_ratio, args.eval_ratio)
    train_loader, eval_loader = dataset_manager.split_and_load(
        train_samples_size=train_split,
        eval_samples_size=eval_split,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers
    )

    # Optimizer
    # TODO doing nothing right now....
    if args.optimizer == "adamw": optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam": optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd": optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum": optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else: raise ValueError("Unknown optimizer {0}".format(args.optimizer))

    # Model
    model = args.model(args.model_config)
    model = model.to(args.device)

    # Print parameters
    if args.verbose :
        print("=="*20)
        for k, v in vars(args).items(): print(k, ":", v)
        print("checkpoint_path:", checkpoint_path)
        print("dataset_size:", len(dataset_manager.dataset))
        print("=="*20)

    # Train    
    model.fit(train_loader, eval_loader)
    plot_training_result(model, len(train_loader.dataset), len(eval_loader.dataset))
    
    return model.all_metrics, checkpoint_path

def train_models(args, seeds:list=[0, 42]):
    """Train a model with different seeds and plot the loss and accuracies of each separately."""
    assert seeds is not None, "Either M or seeds should be provided."

    print(f"Training model {args.exp_name}")
    run_paths = []
    for m, seed in enumerate(seeds):
        print(f"({m+1}/{len(seeds)}) Running training for seed {seed}")
        args.exp_id = m # Set the experiment id
        args.seed = seed # Set the seed
        _, run_path = train_model(args)
        run_paths.append(run_path)

    return run_paths