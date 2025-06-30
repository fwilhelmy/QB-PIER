from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from data.DatasetManager import DatasetManager


class BaseAttack(ABC):
    """Minimal base class used by all adversarial attacks."""

    attack_name: str

    def __init__(
        self,
        attack_name: str,
        dataset,
        max_samples: Optional[int] = None,
    ) -> None:
        self.attack_name = attack_name
        self.dataset = dataset
        self.max_samples = max_samples

    @abstractmethod
    def attack(self, model, epsilon: float, **kwargs) -> Tensor:
        """Generate adversarial examples for ``model`` at ``epsilon`` level."""
        raise NotImplementedError(
            "This method should be overridden in subclasses"
        )
