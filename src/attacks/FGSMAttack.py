from __future__ import annotations

import copy
from typing import List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from src.attacks.BaseAttack import BaseAttack
from src.data.DatasetManager import DatasetManager

class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def __init__(self, dataset_manager: DatasetManager, max_samples: Optional[int] = None) -> None:
        super().__init__("FGSM", dataset_manager, max_samples)

    def _fgsm(self, image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
        if epsilon == 0: return image
        perturbed = image + epsilon * gradient.sign()
        return torch.clamp(perturbed, 0, 1)
    
    @torch.no_grad()
    def compute_gradients(self, model, max_x: Optional[int] = None):
        """Return gradients for ``model`` on this dataset.

        Gradients are computed once and cached unless ``hot_reload`` is True.
        ``max_x`` limits the number of samples used when computing them.
        """
        
        if model.model_name in self.gradients:
            return self.gradients[model.model_name]

        gradients: List[tuple[Tensor, Tensor]] = []
        model.eval()
        state = copy.deepcopy(model.state_dict()) # state-reload for model state
        max_x = max_x if max_x is not None else len(self.dataset)
        pbar = tqdm(self.dataset, desc=f"Computing gradients for {model.model_name}")
        for idx, (image, target) in enumerate(pbar):
            img = image.clone().detach().to(torch.float32).requires_grad_(True)
            output = model(img.unsqueeze(0))
            loss = model.loss_fn(output, torch.tensor([target], dtype=torch.long))
            model.zero_grad()
            loss.backward()
            gradients.append((image, img.grad.detach()))
            model.load_state_dict(state) # state-reload for model state
            if idx >= max_x - 1:
                break

        self.gradients[model.model_name] = gradients
        return gradients

    def attack(self, model, epsilon: float) -> Tensor:
        gradients = self.get_gradients(model)
        adv_examples = []
        for image, gradient in gradients:
            perturbed_image = self._fgsm(image, gradient, epsilon)
            adv_examples.append(perturbed_image)
        return torch.stack(adv_examples)
