from __future__ import annotations

from typing import Dict, List
import copy

import torch
from torch import Tensor
from tqdm import tqdm

from .BaseAttack import BaseAttack
from src.data.DatasetManager import DatasetManager

class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def __init__(self) -> None:
        super().__init__("FGSM")

    def _fgsm(self, image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
        if epsilon == 0: return image
        perturbed = image + epsilon * gradient.sign()
        return torch.clamp(perturbed, 0, 1)

    def attack(self, data, epsilon: float) -> Tensor:
        adv_examples = []
        for image, gradient in data:
            perturbed_image = self._fgsm(image, gradient, epsilon)
            adv_examples.append(perturbed_image)
        return torch.stack(adv_examples)
