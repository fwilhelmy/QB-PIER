"""Factory for :class:`DatasetManager` classes using :class:`HydraFactory`."""

from typing import Callable, Dict, Type

from HydraFactory import HydraFactory

from .DatasetManager import DatasetManager


class DatasetManagerFactory(HydraFactory[DatasetManager]):
    """Registry and entry point for all :class:`DatasetManager` subclasses."""

    _registry: Dict[str, Type[DatasetManager]] = {}

    @classmethod
    def dataset(
        cls, name: str
    ) -> Callable[[Type[DatasetManager]], Type[DatasetManager]]:
        """Decorator to register a :class:`DatasetManager` subclass under ``name``."""

        return super().item(name)

