"""Saliency and perturbation analyticss."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.utilities import rank_zero_only

from attacks.FGSM import FGSM

__all__ = ["old_sal"]

@rank_zero_only
def old_sal(
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
        saliency = grads[i][1].abs().cpu().detach().numpy()
        perturb = attack.attack(model, epsilon).to(device)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(x[i].cpu().squeeze(), cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(saliency.squeeze(), cmap="hot")
        axes[1].set_title("Saliency")
        axes[1].axis("off")
        axes[2].imshow((x[i].cpu() + perturb[i]).squeeze().clamp(0, 1).detach().numpy(), cmap="gray")
        axes[2].set_title(f"FGSM ε={epsilon:.2f}")
        axes[2].axis("off")
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"old_sal_{i}.png"))
        plt.close(fig)
