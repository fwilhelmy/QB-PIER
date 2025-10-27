from typing import Callable, Dict, Type, Optional, Any
from HydraFactory import HydraFactory
from . import BaseModel


class ModelFactory(HydraFactory[BaseModel]):
    """Registry and entry point for all :class:`BaseModel` subclasses."""

    @classmethod
    def model(cls, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator to register a :class:`BaseModel` subclass under ``name``."""

        return super().item(name)

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> BaseModel:
        """Instantiate and optionally configure a registered model."""
        model_cls = cls._get(name)
        model = model_cls(**kwargs)
        # Ensure directories are created for logging
        model._init_directories()
        model.metrics._init_directories()
        return model

    @classmethod
    def load(
        cls,
        name: str,
        checkpoint: str,
        *,
        map_location: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Load a model from ``checkpoint`` using its ``load_from_checkpoint`` method."""
        model_cls = cls._get(name)
        load_fn = getattr(model_cls, "load_from_checkpoint")
        return load_fn(checkpoint, map_location=map_location, **kwargs)


