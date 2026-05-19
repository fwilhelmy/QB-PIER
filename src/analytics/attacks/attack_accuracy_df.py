"""Accuracy vs epsilon analyticss."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.utilities import rank_zero_only

__all__ = ["plot_attack_accuracy_df"]


@rank_zero_only
def plot_attack_accuracy_df(df: pd.DataFrame, source_model: str, root_dir: str):
    sub = df[df["source_model"] == source_model]
    summary = (
        sub.groupby(["epsilon", "target_model"])["accuracy"].agg(["mean", "std"]).reset_index()
    )
    plt.figure(figsize=(6, 4))
    for tgt in summary["target_model"].unique():
        d = summary[summary["target_model"] == tgt]
        plt.errorbar(
            d["epsilon"], d["mean"], yerr=d["std"], marker="o", capsize=3, label=tgt
        )
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(f"{source_model}: Accuracy vs. Epsilon")
    plt.grid(True)
    plt.legend(title="Target Model", fontsize=8)
    plt.tight_layout()

    out_dir = os.path.join(root_dir, source_model)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "plot_attack_accuracy_df.png"))
    plt.close()
