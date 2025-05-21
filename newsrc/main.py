import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import math


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
    train_loader, eval_loader = dataset_manager.split_dataset(args)

    # Optimizer
    if args.optimizer == "adamw": optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam": optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd": optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum": optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else: raise ValueError("Unknown optimizer {0}".format(args.optimizer))

    # Model
    # TODO RESET the MODEL PARAMS WONT WORK WITH MULTI SEEDS RIGHT NOW
    model = model.to(args.device)

    # Print parameters
    if args.verbose :
        print("=="*20)
        for k, v in vars(args).items(): print(k, ":", v)
        print("checkpoint_path:", checkpoint_path)
        print("dataset_size:", len(dataset_manager.dataset))
        print("=="*20)

    # Train    
    all_metrics = train(model, args, checkpoint_path, optimizer, train_loader, eval_loader)
    
    return all_metrics, checkpoint_path

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