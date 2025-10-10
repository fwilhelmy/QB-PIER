from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset



class BaseAttack(ABC):
    """Minimal base class used by all adversarial attacks."""

    attack_name: str

    def __init__(
        self,
        attack_name: str,
        dataset: Dataset,
        max_samples: Optional[int] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        self.attack_name = attack_name
        self.dataset = dataset
        self.max_samples = max_samples or len(self.dataset)
        self.batch_size = batch_size

    @abstractmethod
    def attack(self, model, epsilon: float, **kwargs) -> Tensor:
        """Generate adversarial examples for ``model`` at ``epsilon`` level."""
        raise NotImplementedError(
            "This method should be overridden in subclasses"
        )
