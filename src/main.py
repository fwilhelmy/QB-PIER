import argparse
import torch
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Any

from visualisation.AttacksVisualisation import (
    plot_transfer_attack_effects,
)
from visualisation.TrainingVisualisation import (
    plot_models_together,
    plot_training_curves,
    generate_gradient_flow_plots,
)
from arguments import Arguments
from models.hybrid import (
    AmpVQC,
)

from models.quantum.ansatzs import (
    StrongEntanglingAnsatz
)

import json
import time

from utils import seed_experiment, calculate_accuracy
from data.DatasetManagerFactory import DatasetManagerFactory
from attacks import FGSMAttack
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
    seed_times = {}

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
            print(f" rootdir      : {model.log_dir}")
            print(f" train_size: {len(train_loader.dataset)}")
            print(f" eval_size: {len(eval_loader.dataset)}")
            print(f"accelerator: {args.accelerator}")
            print(f"devices: {args.devices}")
            print(f"num_workers: {args.num_workers}")
            print(f"num_threads: {args.num_threads}")
            print(f"device: {args.device}")
            print("=" * 40)

        # 5) Training
        start_time = time.time()
        model.fit(train_loader, eval_loader)
        elapsed = time.time() - start_time
        seed_times[seed] = elapsed
        trained[seed] = model

    first_model = trained[seeds[0]]
    model_dir = f"{first_model.__class__.__name__}/{str(first_model.model_name)}"

    avg_time = sum(seed_times.values()) / len(seed_times) if seed_times else 0.0
    performance = {"seed_times": seed_times, "average": avg_time}
    LoggingService().save_file(
        content=json.dumps(performance, indent=2),
        title=f"performance.json",
        log_dir=model_dir)
    
    summary: Dict[str, Any] = {
        k: str(v)
        for k, v in vars(args).items()
        if k.startswith("_") is False and k != "seed"
    }
    summary['seeds'] = seeds
    total = 0
    layers_params = {}
    for name, param in first_model.named_parameters():
        n = param.numel()
        layers_params[name] = n
        total += n
    layers_params['total'] = total
    summary['layers_params'] = layers_params
    summary['dataset'] = {
        "name": args.dataset,
        "dataset_size": len(dm.dataset),
        "train_size": len(train_loader.dataset),
        "eval_size": len(eval_loader.dataset),
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
    }
    LoggingService().save_file(
        content=json.dumps(summary, indent=2),
        title=f"summary.json",
        log_dir=model_dir)

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
        
        # Prepare dataset and gradients once for efficiency
        dm = DatasetManagerFactory(args.dataset)
        dataset = dm.load_dataset(train=False)
        dm.dataset = dataset
        data = dataset.data.unsqueeze(1).to(torch.float)
        targets = dataset.targets

        attack = FGSMAttack(dm.dataset)
        epsilons = [x / 100.0 for x in range(0, 102, 2)]

        for attacker_model in models:
            attacker_model.eval()
            for victim_model in models:
                victim_model.eval()
                attack_results: List[Dict[str, float]] = []
                for eps in epsilons:
                    adv_examples = attack.attack(attacker_model, eps).to(args.device)
                    with torch.no_grad():
                        adv_logits = victim_model(adv_examples)
                        adv_acc = calculate_accuracy(
                            adv_logits,
                            targets[: adv_examples.size(0)],
                        )
                    attack_results.append({"epsilon": eps, "accuracy": float(adv_acc)})

                for _, row in pd.DataFrame(attack_results).iterrows():
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
    args.verbose = False
    args.n_steps = 10000
    
    if getattr(args, "num_threads", None) is not None:
        torch.set_num_threads(args.num_threads)

    args.train_ratio = 0.6
    args.eval_ratio = 0.4

    args.train_batch_size = 2**8
    args.eval_batch_size = 2**11

    args.train_log_interval = 1.0
    args.eval_log_interval = 1.0

    # 2) Define your list of model‐configurations: (name, class, kwargs)
    model_specs = [
        # ("AmpVQC_Basic", AmpVQC, {
        #     "input_size": 28 * 28,
        #     "n_qdepth": 4,
        #     "ansatz_cls": BasicEntanglingAnsatz,
        ("AmpVQC_Strong_depth_3", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 3,
            "ansatz_cls": StrongEntanglingAnsatz,
        }),
        ("AmpVQC_Strong_depth_4", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 4,
            "ansatz_cls": StrongEntanglingAnsatz,
        }),
        ("AmpVQC_Strong_depth_5", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 5,
            "ansatz_cls": StrongEntanglingAnsatz,
        }),
        ("AmpVQC_Strong_depth_6", AmpVQC, {
            "input_size": 28 * 28,
            "n_qdepth": 6,
            "ansatz_cls": StrongEntanglingAnsatz,
        }),
        # ("AmpVQC_Random", AmpVQC, {
        #     "input_size": 28 * 28,
        #     "n_qdepth": 4,
        #     "ansatz_cls": RandomStructureAnsatz,
        # }),
        # ("AmpVQC_NoEntangling", AmpVQC, {
        #     "input_size": 28 * 28,
        #     "n_qdepth": 4,
        #     "ansatz_cls": NoEntanglingAnsatz,
        # }),
        # ("AmpVQC_Residual_Basic", AmpVQC, {
        #     "input_size": 28 * 28,
        #     "n_qdepth": 2,
        #     "ansatz_cls": ResidualAnsatzWrapper(BasicEntanglingAnsatz),
        # }),
        # ("AmpVQC_Residual_Strong", AmpVQC, {
        #     "input_size": 28 * 28,
        #     "n_qdepth": 2,
        #     "ansatz_cls": ResidualAnsatzWrapper(StrongEntanglingAnsatz),
        # }),
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
    attacks_metrics = run_attacks(trained_models, args)
    plot_transfer_attack_effects(attacks_metrics,
                                title_prefix="transfer")
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