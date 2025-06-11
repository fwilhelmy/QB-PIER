from __future__ import annotations
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """Minimal base class used by all adversarial attacks."""

    attack_name: str

    def __init__(self, attack_name: str) -> None:
        self.attack_name = attack_name

    @abstractmethod
    def attack(self, model, epsilon):
        """Generate adversarial examples for ``model`` at ``epsilon`` level."""
