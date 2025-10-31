from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import Dataset



class BaseAttack(ABC):
    """Minimal base class used by all adversarial attacks."""

    attack_name: str

    def __init__(
        self,
        attack_name: str,
        dataset: Dataset,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        self.attack_name = attack_name
        self.dataset = dataset
        self.batch_size = batch_size

    @abstractmethod
    def attack(self, model, epsilon: float, **kwargs) -> Tensor:
        """Generate adversarial examples for ``model`` at ``epsilon`` level."""
        raise NotImplementedError(
            "This method should be overridden in subclasses"
        )
