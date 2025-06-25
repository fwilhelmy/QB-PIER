from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from src.data.DatasetManager import DatasetManager


class BaseAttack(ABC):
    """Minimal base class used by all adversarial attacks."""

    attack_name: str

    def __init__(
        self,
        attack_name: str,
        dataset_manager: DatasetManager,
        max_samples: Optional[int] = None,
    ) -> None:
        self.attack_name = attack_name
        self.dataset_manager = dataset_manager
        self.max_samples = max_samples

    @property
    def dataset(self):
        return self.dataset_manager.dataset


    def get_gradients(self, model):
        """Retrieve gradients for ``model`` from the dataset manager."""
        return self.dataset_manager.compute_gradients(model, self.max_samples)

    @abstractmethod
    def attack(self, model, epsilon: float) -> Tensor:
        """Generate adversarial examples for ``model`` at ``epsilon`` level."""
        raise NotImplementedError(
            "This method should be overridden in subclasses"
        )
