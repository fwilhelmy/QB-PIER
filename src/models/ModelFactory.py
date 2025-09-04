from typing import Any, Callable, Dict, Optional, Type

from HydraFactory import HydraFactory

from .BaseModel import BaseModel


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
        *,
        configure: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Instantiate and optionally configure a registered model."""
        model_cls = cls._get(name)
        model = model_cls(**kwargs)
        if configure is not None and hasattr(model, "configure_training"):
            model.configure_training(configure)  # type: ignore[call-arg]
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
        if not hasattr(model_cls, "load_from_checkpoint"):
            raise AttributeError(
                f"{model_cls.__name__} has no 'load_from_checkpoint' method"
            )
        load_fn = getattr(model_cls, "load_from_checkpoint")
        return load_fn(checkpoint, map_location=map_location, **kwargs)


