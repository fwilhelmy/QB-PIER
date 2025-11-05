import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from lightning.pytorch.utilities import rank_zero_only
from LoggingService import LoggingService
from visualisation.training import line_with_std

@rank_zero_only
def plot_transfer_attack_effects(df: pd.DataFrame, title_prefix: str = "transfer"):
    """Plot transfer-attack accuracy curves aggregated across seeds.

    The :func:`~main.run_attacks` helper returns a DataFrame where each
    row corresponds to a seed, a source model and a target model.  The
    remaining columns represent the accuracy of the target model for the
    various ``epsilon`` values that were evaluated.  For backwards
    compatibility, a *long* format with explicit ``epsilon`` and
    ``accuracy`` columns is also accepted.
    
    Parameters
    ----------
    df:
        DataFrame produced by :func:`run_attacks`.
    title_prefix:
        Prefix for figure titles and filenames.
    """

    # ``run_attacks`` now returns a wide DataFrame (one column per epsilon).
    # If we already receive a long/tidy DataFrame, skip this step.
    if "epsilon" not in df.columns:
        id_cols = ["seed", "source_model", "target_model"]
        eps_cols = [c for c in df.columns if c not in id_cols]
        df = df.melt(
            id_vars=id_cols,
            value_vars=eps_cols,
            var_name="epsilon",
            value_name="accuracy",
        )
        df["epsilon"] = df["epsilon"].astype(float)

    attacker_names = sorted(df["source_model"].unique())

    for attacker in attacker_names:
        subset = df[df["source_model"] == attacker]
        victim_names = sorted(subset["target_model"].unique())
        epsilons = sorted(subset["epsilon"].unique())

        plt.figure(figsize=(8, 5))
        for victim in victim_names:
            rec = subset[subset["target_model"] == victim]
            stats = (
                rec.groupby("epsilon")["accuracy"].agg(["mean", "std"]).reindex(epsilons)
            )
            line_with_std(
                plt.gca(), epsilons, stats["mean"], stats["std"], label=victim
            )

        attacker_name = attacker.split("/")[0]
        plt.title(f"Attack by {attacker_name}")
        plt.xlabel("ε (attack strength)")
        plt.ylabel("Accuracy on adversarial examples")
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(title="Victim model", loc="best")

        LoggingService().save_figure(plt.gcf(), f"{title_prefix}{attacker_name}")

@rank_zero_only
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

    LoggingService().save_figure(plt.gcf(), figstr)

@rank_zero_only
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


@rank_zero_only
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


@rank_zero_only
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

from attacks.FGSM import FGSM

@rank_zero_only
def plot_perturbation_and_saliency(
    model,
    attack: FGSM,
    device: torch.device,
    epsilon: float
):
    """
    For a small batch, compute FGSM perturbation and saliency maps,
    then save original, saliency heatmap, and perturbed images side by side.
    """
    model.eval()
    x = attack.dataset.data.to(device)
    grads = torch.stack(attack.gradients[model.model_name])
    saliency = grads.abs().cpu()
    perturb = attack.attack(x, epsilon).to(device)

    out_dir = os.path.join(model.root_dir, "saliency")
    os.makedirs(out_dir, exist_ok=True)

    B = 10
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

@rank_zero_only
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



