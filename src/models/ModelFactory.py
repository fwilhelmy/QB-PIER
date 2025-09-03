from typing import Any, Callable, Dict, List, Optional, Type

from .BaseModel import BaseModel


class ModelFactory:
    """Registry and entry point for all :class:`BaseModel` subclasses."""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseModel]) -> None:
        """Register ``model_cls`` under ``name``."""
        key = name.lower()
        cls._registry[key] = model_cls

    @classmethod
    def model(cls, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator to register a :class:`BaseModel` subclass under ``name``."""

        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            cls.register(name, model_cls)
            return model_cls

        return decorator

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def _get(cls, name: str) -> Type[BaseModel]:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(cls.available())
            raise KeyError(f"Unknown model '{name}'. Available models: {available}")
        return cls._registry[key]

    @classmethod
    def create(
        cls,
        name: str,
        *,
        configure: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Instantiate a new model and optionally configure it for training."""
        model_cls = cls._get(name)
        model = model_cls(**kwargs)
        if configure is not None:
            model.configure_training(configure)
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
        """Load a model from ``checkpoint`` using Lightning's loader."""
        model_cls = cls._get(name)
        return model_cls.load_from_checkpoint(
            checkpoint, map_location=map_location, **kwargs
        )


