"""Factory for :class:`DatasetManager` classes using :class:`HydraFactory`."""

from typing import Callable, Dict, Type

from HydraFactory import HydraFactory
from .BaseDataset import BaseDataset


class DatasetFactory(HydraFactory[BaseDataset]):
    """Registry and entry point for all :class:`DatasetManager` subclasses."""

    _registry: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def dataset(
        cls, name: str
    ) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:
        """Decorator to register a :class:`DatasetManager` subclass under ``name``."""

        return super().item(name)

