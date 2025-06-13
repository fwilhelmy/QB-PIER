from __future__ import annotations

from typing import Dict, List
import copy

import torch
from torch import Tensor
from tqdm import tqdm

from models.BaseModel import BaseModel
from .BaseAttack import BaseAttack


class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def __init__(self, dataset) -> None:
        super().__init__("FGSM")
        self.dataset = dataset
        self.gradients: Dict[str, List[Tensor]] = {}

    def _fgsm(self, image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
        if epsilon == 0: return image
        perturbed = image + epsilon * gradient.sign()
        return torch.clamp(perturbed, 0, 1)

    def _calculate_gradient(self, model, image: Tensor, target: int) -> Tensor:
        img = image.clone().detach().to(torch.float32).requires_grad_(True)
        output = model(img.unsqueeze(0))
        loss = model.loss_fn(output, torch.tensor([target], dtype=torch.long))
        model.zero_grad()
        loss.backward()
        return img.grad.detach()

    def _precompute_gradients(self, model) -> None:
        gradients: List[Tensor] = []
        model.eval()
        state = copy.deepcopy(model.state_dict())
        pbar = tqdm(self.dataset, desc=f"Computing gradients for {model.model_name}")
        for idx, (img, target) in enumerate(pbar):
            gradients.append(self._calculate_gradient(model, img, target))
            model.load_state_dict(state)
            if idx >= 149:
                break
        self.gradients[model.model_name] = gradients

    def attack(self, model, epsilon: float) -> Tensor:
        if model.model_name not in self.gradients:
            self._precompute_gradients(model)

        adv_examples = []
        for idx, (img, _) in enumerate(self.dataset):
            gradient = self.gradients[model.model_name][idx]
            adv_examples.append(self._fgsm(img, gradient, epsilon))
            if idx >= 149:
                break

        return torch.stack(adv_examples)
