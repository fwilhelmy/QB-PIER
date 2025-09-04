from __future__ import annotations

import copy
from typing import List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from torch.utils.data import Dataset

from attacks.BaseAttack import BaseAttack

# TODO Test model.freeze()

class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def __init__(self, dataset: Dataset, max_samples: Optional[int] = None) -> None:
        super().__init__("FGSM", dataset, max_samples)
        self.gradients: dict[str, List[tuple[Tensor, Tensor]]] = {}

    def _fgsm(self, image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
        if epsilon == 0: return image
        perturbed = image + epsilon * gradient.sign()
        return torch.clamp(perturbed, 0, 1)
    
    def compute_gradients(
        self,
        model,
        max_x: Optional[int] = None,
        batch_size: int = 64,
    ):
        """Return gradients for ``model`` on this dataset.

        Gradients are computed once and cached unless ``hot_reload`` is True.
        ``max_x`` limits the number of samples used when computing them.
        ``batch_size`` controls how many samples are processed at once.
        """

        key = getattr(model, "model_path", getattr(model, "model_name", str(id(model))))
        if key in self.gradients:
            return self.gradients[key]

        gradients: List[tuple[Tensor, Tensor]] = []
        model.eval()
        state = copy.deepcopy(model.state_dict())  # save model weights
        max_x = max_x or self.max_samples or len(self.dataset)

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
        )
        processed = 0
        pbar = tqdm(loader, desc=f"Computing gradients for {key}")
        for images, targets in pbar:
            if processed >= max_x:
                break
            images = images.clone().detach().to(torch.float32)
            images.requires_grad_(True)
            outputs = model(images)
            loss = model.loss_fn(outputs, targets)
            model.zero_grad()
            loss.backward()
            grads = images.grad.detach()
            for img, grad in zip(images, grads):
                gradients.append((img.cpu(), grad.cpu()))
                processed += 1
                if processed >= max_x:
                    break
            model.load_state_dict(state)

        self.gradients[key] = gradients
        return gradients

    def attack(self, model, epsilon: float, **kwargs) -> Tensor:
        gradients = self.compute_gradients(model, max_x=self.max_samples)
        adv_examples = []
        for image, gradient in gradients:
            perturbed_image = self._fgsm(image, gradient, epsilon)
            adv_examples.append(perturbed_image)
        return torch.stack(adv_examples)
