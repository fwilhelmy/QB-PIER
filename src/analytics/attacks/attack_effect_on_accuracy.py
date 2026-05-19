"""Attack strength vs accuracy analyticss."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only

__all__ = ["plot_attack_effect_on_accuracy"]


@rank_zero_only
def plot_attack_effect_on_accuracy(figstr, models, epsilons, accuracies, out_dir):
    plt.figure(figsize=(10, 6))

    for model in models:
        plt.plot(epsilons, accuracies[model.log_dir], label=model.log_dir, marker="o")

    plt.xlabel("Attack Strength (Epsilon)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Effect of Adversarial Attacks on Model Accuracy", fontsize=14)
    plt.xticks(epsilons, [f"{e:.2f}" for e in epsilons], rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Models", fontsize=10)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, figstr + ".png"))
    plt.close()
