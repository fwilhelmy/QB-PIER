from __future__ import annotations

from typing import Optional

import copy

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from attacks.BaseAttack import BaseAttack


class PGDAttack(BaseAttack):
    """Projected Gradient Descent adversarial attack."""

    def __init__(
        self,
        dataset: Dataset,
        num_iter: int = 10,
        alpha: float = 0.01,
        random_start: bool = True,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__("PGD", dataset, max_samples)
        self.num_iter = num_iter
        self.alpha = alpha
        self.random_start = random_start
        self.batch_size = batch_size

    def _pgd(self, model, images: Tensor, labels: Tensor, epsilon: float) -> Tensor:
        """Apply PGD to a batch of images."""
        if self.random_start:
            delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        else:
            delta = torch.zeros_like(images)
        delta.requires_grad_(True)

        for _ in range(self.num_iter):
            adv_images = torch.clamp(images + delta, 0.0, 1.0)
            outputs = model(adv_images)
            loss = model.loss_fn(outputs, labels)
            model.zero_grad()
            loss.backward()
            grad = delta.grad.detach()
            with torch.no_grad():
                delta.add_(self.alpha * grad.sign()).clamp_(-epsilon, epsilon)
            delta.grad.zero_()

        adv_images = torch.clamp(images + delta.detach(), 0.0, 1.0)
        return adv_images

    def attack(self, model, epsilon: float, **kwargs) -> Tensor:
        """Generate PGD adversarial examples for ``model`` on this attack's dataset."""
        adv_batches = []
        device = next(model.parameters()).device
        model.eval()
        state = copy.deepcopy(model.state_dict())  # save original weights
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        for idx, (images, labels) in enumerate(loader):
            if self.max_samples is not None and idx >= self.max_samples:
                break
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device)
            adv = self._pgd(model, images, labels, epsilon)
            adv_batches.append(adv.detach().cpu())
            model.load_state_dict(state)  # restore weights after PGD
        return torch.cat(adv_batches)

