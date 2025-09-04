from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


class HydraFactory(Generic[T]):
    """Generic registry and factory for subclasses of a given base class."""

    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, name: str, obj_cls: Type[T]) -> None:
        """Register ``obj_cls`` under ``name``."""
        key = name.lower()
        cls._registry[key] = obj_cls

    @classmethod
    def item(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a subclass under ``name``."""

        def decorator(obj_cls: Type[T]) -> Type[T]:
            cls.register(name, obj_cls)
            return obj_cls

        return decorator

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of registered names."""
        return sorted(cls._registry.keys())

    @classmethod
    def _get(cls, name: str) -> Type[T]:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(cls.available())
            raise KeyError(
                f"Unknown {cls.__name__[:-7].lower()} '{name}'. Available: {available}"
            )
        return cls._registry[key]

    @classmethod
    def create(
        cls,
        name: str,
        *,
        configure: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """Instantiate a registered class and optionally configure it."""
        obj_cls = cls._get(name)
        obj = obj_cls(**kwargs)
        if configure is not None and hasattr(obj, "configure_training"):
            obj.configure_training(configure)  # type: ignore[call-arg]
        return obj

    @classmethod
    def load(
        cls,
        name: str,
        checkpoint: str,
        *,
        map_location: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Load an object from ``checkpoint`` using its ``load_from_checkpoint`` method."""
        obj_cls = cls._get(name)
        if not hasattr(obj_cls, "load_from_checkpoint"):
            raise AttributeError(
                f"{obj_cls.__name__} has no 'load_from_checkpoint' method"
            )
        load_fn = getattr(obj_cls, "load_from_checkpoint")
        return load_fn(checkpoint, map_location=map_location, **kwargs)
