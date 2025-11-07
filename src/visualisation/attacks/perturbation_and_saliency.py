"""Saliency and perturbation visualisations."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities import rank_zero_only

from attacks.FGSM import FGSM

__all__ = ["plot_perturbation_and_saliency"]


@rank_zero_only
def plot_perturbation_and_saliency(
    model,
    attack: FGSM,
    device: torch.device,
    epsilon: float,
    out_dir,
):
    model.eval()
    x = attack.dataset.data.to(device)
    grads = attack.gradients[model.log_dir]
    B = 3
    for i in range(B):
        image = x[i].cpu().squeeze().detach().numpy()
        saliency = grads[i][1].detach().cpu()
        saliency_sign = saliency.sign().numpy()

        # Normalise saliency for visualising overlay
        saliency_abs = saliency.abs()
        if saliency_abs.max() > 0:
            saliency_overlay = (saliency_abs / saliency_abs.max()).numpy()
        else:
            saliency_overlay = saliency_abs.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(saliency_sign.squeeze(), cmap="hot")
        axes[1].set_title("Saliency")
        axes[1].axis("off")
        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(saliency_overlay.squeeze(), cmap="viridis", alpha=0.6)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"saliency_{i}.png"))
        plt.close(fig)
