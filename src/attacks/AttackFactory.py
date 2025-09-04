from typing import Callable, Type

from HydraFactory import HydraFactory

from .BaseAttack import BaseAttack


class AttackFactory(HydraFactory[BaseAttack]):
    """Registry and entry point for all :class:`BaseAttack` subclasses."""

    @classmethod
    def attack(cls, name: str) -> Callable[[Type[BaseAttack]], Type[BaseAttack]]:
        """Decorator to register a :class:`BaseAttack` subclass under ``name``."""

        return super().item(name)
