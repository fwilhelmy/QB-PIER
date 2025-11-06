from __future__ import annotations

import copy
from typing import List

import torch
from torch import Tensor
from tqdm import tqdm

from attacks import AttackFactory, BaseAttack

# TODO Test model.freeze()


@AttackFactory.attack("fgsm")
class FGSM(BaseAttack):
    """Fast Gradient Sign Method attack."""

    def __init__(self, **kwargs) -> None:
        super().__init__("FGSM", **kwargs)
        self.gradients: dict[str, List[tuple[Tensor, Tensor]]] = {}

    def _fgsm(self, image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
        if epsilon == 0: return image
        perturbed = image + epsilon * gradient.sign()
        return torch.clamp(perturbed, 0, 1)
    
    def compute_gradients(
        self,
        model,
    ):
        """Return gradients for ``model`` on this dataset.

        Gradients are computed once and cached unless ``hot_reload`` is True.
        ``batch_size`` controls how many samples are processed at once.
        """

        if model.log_dir in self.gradients:
            return self.gradients[model.log_dir]

        gradients: List[tuple[Tensor, Tensor]] = []
        model.eval()
        state = copy.deepcopy(model.state_dict())  # save model weights

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
        pbar = tqdm(loader, desc=f"Computing gradients")
        for images, targets in pbar:
            images = images.clone().detach().to(torch.float32)
            images.requires_grad_(True)
            outputs = model(images)
            loss = model.loss_fn(outputs, targets)
            model.zero_grad()
            loss.backward()
            grads = images.grad.detach()
            for img, grad in zip(images, grads):
                gradients.append((img.cpu(), grad.cpu()))
            model.load_state_dict(state)

        self.gradients[model.log_dir] = gradients
        return gradients

    def attack(self, model, epsilon: float, **kwargs) -> Tensor:
        gradients = self.compute_gradients(model)
        adv_examples = []
        for image, gradient in gradients:
            perturbed_image = self._fgsm(image, gradient, epsilon)
            adv_examples.append(perturbed_image)
        return torch.stack(adv_examples)
