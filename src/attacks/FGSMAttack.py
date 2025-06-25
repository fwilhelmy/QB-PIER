from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .BaseAttack import BaseAttack
from src.data.DatasetManager import DatasetManager

class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def __init__(self, dataset_manager: DatasetManager, max_samples: Optional[int] = None) -> None:
        super().__init__("FGSM", dataset_manager, max_samples)

    def _fgsm(self, image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
        if epsilon == 0: return image
        perturbed = image + epsilon * gradient.sign()
        return torch.clamp(perturbed, 0, 1)

    def attack(self, model, epsilon: float) -> Tensor:
        gradients = self.get_gradients(model)
        adv_examples = []
        for image, gradient in gradients:
            perturbed_image = self._fgsm(image, gradient, epsilon)
            adv_examples.append(perturbed_image)
        return torch.stack(adv_examples)
