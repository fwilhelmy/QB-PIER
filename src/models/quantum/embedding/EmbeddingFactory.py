"""Factory for creating embedding classes using :class:`HydraFactory`."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from HydraFactory import HydraFactory
from .BaseEmbedding import BaseEmbedding


class EmbeddingFactory(HydraFactory[BaseEmbedding]):
    """Registry and factory for :class:`BaseEmbedding` subclasses."""

    _registry: Dict[str, Type[BaseEmbedding]] = {}

    @classmethod
    def embedding(cls, name: str) -> Callable[[Type[BaseEmbedding]], Type[BaseEmbedding]]:
        """Decorator to register an embedding under ``name``."""
        return super().item(name)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        input_size: int,
        configure: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[BaseEmbedding | None, int]:
        """Instantiate an embedding and return it with its qubit count."""
        embedding = super().create(
            name, configure=configure, input_size=input_size, **kwargs
        )
        return embedding, embedding.n_qubits
