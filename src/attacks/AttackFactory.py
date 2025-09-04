from typing import Callable, Dict, Type

from HydraFactory import HydraFactory

from .BaseAttack import BaseAttack


class AttackFactory(HydraFactory[BaseAttack]):
    """Registry and entry point for all :class:`BaseAttack` subclasses."""

    _registry: Dict[str, Type[BaseAttack]] = {}

    @classmethod
    def attack(cls, name: str) -> Callable[[Type[BaseAttack]], Type[BaseAttack]]:
        """Decorator to register a :class:`BaseAttack` subclass under ``name``."""

        return super().item(name)
