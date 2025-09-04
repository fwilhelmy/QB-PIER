from typing import Callable, Dict, Type

from HydraFactory import HydraFactory

from .BaseModel import BaseModel


class ModelFactory(HydraFactory[BaseModel]):
    """Registry and entry point for all :class:`BaseModel` subclasses."""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def model(cls, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator to register a :class:`BaseModel` subclass under ``name``."""

        return super().item(name)


