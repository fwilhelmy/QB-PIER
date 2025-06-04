import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import math
from visualisation.AttacksVisualisation import plot_attack_effect_on_accuracy, visualize_attacks_results
from visualisation.TrainingVisualisation import plot_training_result, plot_models_together
from arguments import Arguments
from models.classical.LeNet import LeNet
from models.classical.mlp import MLP
from models.classical.resnet18 import ResNet18
from models.classical.mlpmixer import MLPMixer
from models.hybrid.HQNN_Quanv import HQNN_Quanv
from models.hybrid.HQNN_Parallel import HQNN_Parallel
from utils import seed_experiment
from data.DatasetManagerFactory import DatasetManagerFactory

import torch

def train_model(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

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

    # Model
    model = args.model(
        model_name=args.exp_name,
        seed=args.seed,
        **args.model_config)
    # TODO add the other training hyperparameters
    model.configure_training(
        optimizer=args.optimizer,
        loss_fn=args.loss_fn,
        learning_rate=args.lr,
        max_steps=args.n_steps,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        devices=args.device,
    )
    model = model.to(args.device)

    # Print parameters
    if args.verbose :
        print("=="*20)
        for k, v in vars(args).items(): print(k, ":", v)
        print("rootdir:", model.root_dir)
        print("dataset_size:", len(dataset_manager.dataset))
        print("=="*20)

    # Train
    model.fit(train_loader, eval_loader)
    plot_training_result(model, train_split, eval_split)
    
    return model.all_metrics, model.root_dir

def run_experiment(args, seeds:list=[0, 42]):
    """Train a model with different seeds and plot the loss and accuracies of each separately."""
    assert seeds is not None, "Either M or seeds should be provided."

    print(f"Training model {args.exp_name}")
    run_paths = []
    for m, seed in enumerate(seeds):
        print(f"({m+1}/{len(seeds)}) Running training for seed {seed}")
        args.seed = seed # Set the seed
        _, run_path = train_model(args)
        run_paths.append(run_path)

    return run_paths

if 'main' in __name__:
    seeds = [0, 42]
    experiment1 = Arguments()
    experiment1.verbose = False
    experiment1.n_steps = 100
    experiment1.num_workers = 0
    experiment1.train_batch_size = 512
    experiment1.eval_batch_size = 512

    models = [
        [
            "HQNN_Parallel_1",
            HQNN_Parallel, 
            {'in_channels': 1, 'classes': 10}
        ],
        [
            "LeNet_1",
            LeNet,
            {'in_channels': 1, 'classes': 10}
        ],
    ]

    for model_name, model, config in models:
        experiment1.exp_name = model_name
        experiment1.model = model
        experiment1.model_config = config
        run_experiment(experiment1, seeds)   

    # TODO Should implement some sanity checks, so usefull when testing everything...

        # TODO Quantum MLPMixer

    # TODO Gradient flow visualisation

    # TODO Graph flow of model visualisation

    # TODO add checkpoints

        # TODO What would really be interesting beyond just looking at curves if it resist or not
    # Would be to visualise how the point are placed in the hillbert space where are the frontiers
    # and how attacking a certain VQC is going to change the position of the data points in the hilbert space

        # TODO so basically how I see things, we should have a function you call to setup the
    # models, in it you should pass :
    # - the dataset and dataset configuration (batch size, num_workers, etc..)
    # - the model class
    # - the model configuration
    # - the training/optimization configuration