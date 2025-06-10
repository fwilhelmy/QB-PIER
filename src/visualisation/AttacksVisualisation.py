import os
import sys
# ─── Make sure Python can find `LoggingService.py` in the parent directory ───
# (i.e. <repo_root>/newsrc/LoggingService.py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ─────────────────────────────────────────────────────────────────────────────
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from LoggingService import LoggingService


import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(trained_models, title_prefix="training"):
    """
    Plot averaged training & validation loss and accuracy curves for each model.
    
    Args:
        trained_models (dict):
            Mapping from model_name to { seed: BaseModel_instance }.
        title_prefix (str): 
            Prefix for figure titles and filenames.
        save_dir (str or None):
            Directory in which to save PNG files. If None, figures are shown interactively.
    """
    for model_name, seed_dict in trained_models.items():
        seeds = sorted(seed_dict.keys())

        # Gather per-seed metric lists
        train_loss_seeds = []
        val_loss_seeds   = []
        train_acc_seeds  = []
        val_acc_seeds    = []
        for seed in seeds:
            metrics = seed_dict[seed].all_metrics
            # BaseModel now stores metrics in a nested dict keyed by phase
            train_loss_seeds.append(metrics["train"]["loss"])
            val_loss_seeds.append(metrics["val"]["loss"])
            train_acc_seeds.append(metrics["train"]["acc"])
            val_acc_seeds.append(metrics["val"]["acc"])

        # Truncate to shortest length to align curves
        min_len = min(
            *(len(lst) for lst in train_loss_seeds + val_loss_seeds
                  + train_acc_seeds + val_acc_seeds)
        )
        steps = np.arange(min_len)

        # Stack into arrays of shape (n_seeds, min_len)
        tl = np.vstack([lst[:min_len] for lst in train_loss_seeds])
        vl = np.vstack([lst[:min_len] for lst in val_loss_seeds])
        ta = np.vstack([lst[:min_len] for lst in train_acc_seeds])
        va = np.vstack([lst[:min_len] for lst in val_acc_seeds])

        # Compute mean and std along axis=0 (across seeds)
        tl_mean, tl_std = tl.mean(0), tl.std(0)
        vl_mean, vl_std = vl.mean(0), vl.std(0)
        ta_mean, ta_std = ta.mean(0), ta.std(0)
        va_mean, va_std = va.mean(0), va.std(0)

        # Build figure with two subplots: loss & accuracy
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 10))

        # --- Loss subplot ---
        ax_loss.plot(steps, tl_mean, label="Train Loss")
        ax_loss.fill_between(steps,
                             tl_mean - tl_std,
                             tl_mean + tl_std,
                             alpha=0.2)
        ax_loss.plot(steps, vl_mean, label="Val Loss")
        ax_loss.fill_between(steps,
                             vl_mean - vl_std,
                             vl_mean + vl_std,
                             alpha=0.2)
        ax_loss.set_title(f"{model_name} Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", linewidth=0.5)
        ax_loss.legend()

        # --- Accuracy subplot ---
        ax_acc.plot(steps, ta_mean, label="Train Acc")
        ax_acc.fill_between(steps,
                             ta_mean - ta_std,
                             ta_mean + ta_std,
                             alpha=0.2)
        ax_acc.plot(steps, va_mean, label="Val Acc")
        ax_acc.fill_between(steps,
                             va_mean - va_std,
                             va_mean + va_std,
                             alpha=0.2)
        ax_acc.set_title(f"{model_name} Accuracy")
        ax_acc.set_xlabel("Step")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.0)
        ax_acc.grid(True, linestyle="--", linewidth=0.5)
        ax_acc.legend()

        fig.suptitle(f"{title_prefix.capitalize()} Curves — {model_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        LoggingService().save_figure(plt, f"{title_prefix}_{model_name}")

def plot_transfer_attack_effects(attacks_metrics, title_prefix="transfer"):
    """
    For each attacker model, plot one figure showing transfer‐attack accuracy
    curves on all victim models—averaged over seeds, with ±1 std shaded.

    Args:
        attacks_metrics (dict):
            Mapping: seed → attacker_name → victim_name → {epsilon: accuracy}
        title_prefix (str): prefix for titles and filenames
        save_dir (str or None): if set, directory to save PNGs; else displays plots
    """
    seeds = sorted(attacks_metrics.keys())
    attacker_names = sorted(attacks_metrics[seeds[0]].keys())

    for attacker in attacker_names:
        # get victim list and epsilons from first seed
        victim_names = sorted(attacks_metrics[seeds[0]][attacker].keys())
        epsilons = sorted(next(iter(attacks_metrics[seeds[0]][attacker].values())).keys())

        plt.figure(figsize=(8, 5))
        for victim in victim_names:
            # build accuracy matrix [n_seeds x n_epsilons]
            acc_matrix = np.array([
                [attacks_metrics[seed][attacker][victim][eps] for eps in epsilons]
                for seed in seeds
            ], dtype=float)

            mean_curve = acc_matrix.mean(axis=0)
            std_curve  = acc_matrix.std(axis=0)

            plt.plot(epsilons, mean_curve, label=victim)
            plt.fill_between(epsilons,
                             mean_curve - std_curve,
                             mean_curve + std_curve,
                             alpha=0.2)

        plt.title(f"{title_prefix.capitalize()} attack by {attacker}")
        plt.xlabel("ε (attack strength)")
        plt.ylabel("Accuracy on adversarial examples")
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(title="Victim model", loc="best")

        LoggingService().save_figure(plt, f"{title_prefix}_{attacker}")

def plot_attack_effect_on_accuracy(figstr, models, epsilons, accuracies):
    plt.figure(figsize=(10, 6))

    for model in models:
        plt.plot(epsilons, accuracies[model.model_name], label=model.model_name, marker='o')

    plt.xlabel('Attack Strength (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Effect of Adversarial Attacks on Model Accuracy', fontsize=14)
    plt.xticks(epsilons, [f"{e:.2f}" for e in epsilons], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models', fontsize=10)
    plt.tight_layout()

    LoggingService().save_figure(plt, figstr)

def store_attack_results(records: list, root_dir: str) -> pd.DataFrame:
    """
    Turn a list of dicts into a pandas DataFrame, save to CSV, and return it.
    Each record should have keys: source_model, target_model, seed, epsilon, accuracy.
    """
    df = pd.DataFrame(records)
    os.makedirs(root_dir, exist_ok=True)
    out_path = os.path.join(root_dir, "attack_results.csv")
    df.to_csv(out_path, index=False)
    return df


def plot_attack_accuracy_df(df: pd.DataFrame, source_model: str, root_dir: str):
    """
    For a given source_model, plot accuracy vs epsilon for each target_model,
    with error bars (mean ± std) over all seeds.
    """
    sub = df[df['source_model'] == source_model]
    summary = (
        sub
        .groupby(['epsilon','target_model'])['accuracy']
        .agg(['mean','std'])
        .reset_index()
    )
    plt.figure(figsize=(6, 4))
    for tgt in summary['target_model'].unique():
        d = summary[summary['target_model'] == tgt]
        plt.errorbar(
            d['epsilon'], d['mean'], yerr=d['std'],
            marker='o', capsize=3, label=tgt
        )
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(f"{source_model}: Accuracy vs. Epsilon")
    plt.grid(True)
    plt.legend(title="Target Model", fontsize=8)
    plt.tight_layout()

    out_dir = os.path.join(root_dir, source_model)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "accuracy_vs_epsilon.png"))
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    root_dir: str,
    source_model: str,
    normalize: bool = True
):
    """
    Save a (normalized) confusion matrix for the attacked source_model.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plt.title(f"{source_model} @ max ε: {title}")
    plt.tight_layout()

    out_dir = os.path.join(root_dir, source_model)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


def plot_perturbation_and_saliency(
    model,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    device: torch.device,
    epsilon: float
):
    """
    For a small batch, compute FGSM perturbation and saliency maps,
    then save original, saliency heatmap, and perturbed images side by side.
    """
    model.eval()
    x = x_batch.to(device)
    x.requires_grad_()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y_batch.to(device))
    loss.backward()
    saliency = x.grad.data.abs().cpu()
    perturb = epsilon * x.grad.sign().cpu()

    out_dir = os.path.join(model.root_dir, "saliency")
    os.makedirs(out_dir, exist_ok=True)

    B = x.shape[0]
    for i in range(B):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(x[i].cpu().squeeze(), cmap='gray')
        axes[0].set_title("Original"); axes[0].axis('off')
        axes[1].imshow(saliency[i].squeeze(), cmap='hot')
        axes[1].set_title("Saliency"); axes[1].axis('off')
        axes[2].imshow((x[i].cpu() + perturb[i]).squeeze().clamp(0,1), cmap='gray')
        axes[2].set_title(f"FGSM ε={epsilon:.2f}"); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"saliency_{i}.png"))
        plt.close()


def plot_loss_landscape(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    alphas: np.ndarray,
    betas: np.ndarray,
    device: torch.device
):
    """
    Pick two random orthogonal directions in input space and plot the 2D loss surface
    over (alpha, beta) perturbations around x.
    """
    model.eval()
    x0 = x.to(device)
    # generate two random orthonormal directions
    d1 = torch.randn_like(x0)
    d1 /= torch.norm(d1)
    d2 = torch.randn_like(x0)
    d2 -= (d2 * d1).sum() * d1
    d2 /= torch.norm(d2)

    Z = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            xp = (x0 + a*d1 + b*d2).to(device)
            lp = torch.nn.functional.cross_entropy(model(xp), y.to(device))
            Z[i, j] = lp.item()

    A, B = np.meshgrid(betas, alphas)
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(A, B, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Loss')
    plt.xlabel("β"); plt.ylabel("α")
    plt.title("Loss Landscape")
    plt.tight_layout()

    out_dir = os.path.join(model.root_dir, "loss_landscape")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "loss_landscape.png"))
    plt.close()


def _plot_gradient_flow(grad_norms: dict, title: str, save_path: str):
    layers, norms = zip(*sorted(grad_norms.items(), key=lambda kv: kv[0]))
    plt.figure(figsize=(8, 4))
    for name, vals in zip(layers, norms):
        plt.plot(vals, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title(title)
    plt.legend(fontsize=6, ncol=2, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_gradient_flow_plots(list_of_dirs: list, legend_names: list):
    """
    For each log directory containing a results.json with 'train_grad_norms',
    plot gradient‐flow for weights and biases.
    """
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        if not os.path.exists(json_path):
            raise AssertionError(f"No results.json in {logdir}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        norms = data['train_grad_norms']

        weights = {k: v for k, v in norms.items() if 'weight' in k and 'bias' not in k}
        biases  = {k: v for k, v in norms.items() if 'bias' in k}

        w_path = os.path.join(logdir, f"gradient_flow_{name}_weights.png")
        b_path = os.path.join(logdir, f"gradient_flow_{name}_biases.png")
        _plot_gradient_flow(weights, f"Gradient Flow—Weights ({name})", w_path)
        _plot_gradient_flow(biases,  f"Gradient Flow—Biases ({name})",  b_path)
