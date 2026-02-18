"""Transfer attack accuracy visualisations."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.utilities import rank_zero_only

from visualisation.utils import line_with_std

__all__ = ["plot_transfer_attack_effects"]


@rank_zero_only
def plot_transfer_attack_effects(df: pd.DataFrame, out_dir: str, title_prefix: str = "transfer"):
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
            stats = rec.groupby("epsilon")["accuracy"].agg(["mean", "std"]).reindex(epsilons)
            line_with_std(plt.gca(), epsilons, stats["mean"], stats["std"], label=victim)

        attacker_name = attacker.split("/")[0]
        plt.title(f"Attack by {attacker_name}")
        plt.xlabel("ε (attack strength)")
        plt.ylabel("Accuracy on adversarial examples")
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(title="Victim model", loc="best")

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{title_prefix}{attacker_name}.png"))
        plt.close()
