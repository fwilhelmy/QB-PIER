from __future__ import annotations
from abc import ABC, abstractmethod

from torch import Tensor


class BaseAttack(ABC):
    """Minimal base class used by all adversarial attacks."""

    attack_name: str

    def __init__(self, attack_name: str) -> None:
        self.attack_name = attack_name

    @abstractmethod
    def attack(self, data, epsilon: float) -> Tensor:
        """Generate adversarial examples for ``model`` at ``epsilon`` level."""
        raise NotImplementedError("This method should be overridden in subclasses")
