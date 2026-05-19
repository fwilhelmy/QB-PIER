"""Loss landscape analyticss."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only

__all__ = ["plot_loss_landscape"]


@rank_zero_only
def plot_loss_landscape(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    alphas: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
):
    model.eval()
    x0 = x.to(device)
    d1 = torch.randn_like(x0)
    d1 /= torch.norm(d1)
    d2 = torch.randn_like(x0)
    d2 -= (d2 * d1).sum() * d1
    d2 /= torch.norm(d2)

    Z = np.zeros((len(alphas), len(betas)))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            xp = (x0 + a * d1 + b * d2).to(device)
            lp = torch.nn.functional.cross_entropy(model(xp), y.to(device))
            Z[i, j] = lp.item()

    A, B = np.meshgrid(betas, alphas)
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(A, B, Z, levels=50, cmap="viridis")
    plt.colorbar(cp, label="Loss")
    plt.xlabel("β")
    plt.ylabel("α")
    plt.title("Loss Landscape")
    plt.tight_layout()

    out_dir = os.path.join(model.root_dir, "loss_landscape")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "loss_landscape.png"))
    plt.close()
