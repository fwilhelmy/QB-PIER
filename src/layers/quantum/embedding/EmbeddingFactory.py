"""Factory for creating embedding classes using :class:`HydraFactory`."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from HydraFactory import HydraFactory
from . import BaseEmbedding


class EmbeddingFactory(HydraFactory[BaseEmbedding]):
    """Registry and factory for :class:`BaseEmbedding` subclasses."""

    _registry: Dict[str, Type[BaseEmbedding]] = {}

    @classmethod
    def embedding(cls, name: str) -> Callable[[Type[BaseEmbedding]], Type[BaseEmbedding]]:
        """Decorator to register an embedding under ``name``."""
        return super().item(name)
